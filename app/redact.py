"""敏感字段脱敏引擎（``x-sensitive`` 标记驱动的实例值替换）。

针对一次分析已保存的**新 schema**（``new_schema``），把提交文档中所有被
schema 节点标注 ``x-sensitive: true`` 的实例值整体替换为 ``"***"``，
并报告每一处替换位置的实例指针（JSON Pointer，根为 ``#``）。

遍历规则
========
* **对象属性 / 数组元素**：``properties`` 按实例中实际存在的属性下钻；
  数组支持 ``prefixItems``（2020-12 元组）、列表形式 ``items``
  （draft-04/06/07 元组）与字典形式 ``items``（齐次数组，跳过
  ``prefixItems`` 已覆盖的前缀）；
* **文档内 $ref**：仅解析 ``#`` 与 ``#/...`` 本地指针（RFC 6901，含
  ``~0`` / ``~1`` 转义与百分号编码），解析后跳到目标节点继续遍历；
  与 ``compat.resolve_refs`` 一致采用 draft-07 语义——节点含 ``$ref``
  时其兄弟关键字不再生效。同一实例位置遇到递归引用（``(引用目标,
  实例位置)`` 对已展开过）时停止重复展开，保证 A <-> B 互引、自引
  ``#`` 等循环 schema 上必然终止；
* **组合关键字**：``allOf`` 遍历全部分支；``oneOf`` / ``anyOf`` 只遍历
  对当前实例**校验通过**的分支（分支校验在完整 schema 的 ``$ref`` 包装
  校验器上进行，分支内的 ``#/...`` 引用仍能解析到原始文档）；
* **x-sensitive 命中**：记录实例指针并停止下钻（子树被整体遮蔽）；若更
  深层位置此前已被记录（如经 ``allOf`` 另一分支到达），以祖先位置为准
  ——最终替换位置集合不存在祖先 / 后代重叠。

脱敏分两阶段执行：先在**原始文档上只读**遍历收集替换位置（oneOf/anyOf
分支选择因此始终看到未被遮蔽的原始值），再在深拷贝上统一替换为
``"***"``——不改动调用方传入的原文档，也不修改库中保存的 schema 与样本。

无法解析的 ``$ref``（外部引用、锚点名称片段、目标不存在、非法指针转义、
非字符串取值）由 ``find_invalid_refs`` 预先扫描（复用引用诊断的遍历与
分类），调用方据此返回带 schema 指针的 422。

引擎为纯函数，不读取数据库、不修改入参、不做任何持久化。
"""
from __future__ import annotations

import copy
from typing import Any

from app import refgraph
from app.compat import escape_pointer, make_validator
from app.profile import render_instance_pointer

# 敏感值的统一替换文本。
MASK = "***"

# 单次脱敏允许的最大文档数（与 schemas.RedactRequest 保持一致）。
MAX_REDACT_DOCUMENTS = 5000


def find_invalid_refs(schema: Any) -> list[dict]:
    """预先扫描 schema 中全部无法解析的 ``$ref``（按源指针排序，确定性）。

    返回条目与引用诊断接口一致：source（声明 ``$ref`` 的 schema 指针）、
    ref（原文）、reason（原因代码）与 message。
    """
    edges: list[dict] = []
    invalid: list[dict] = []
    refgraph._walk(schema, "#", schema, edges, invalid)
    invalid.sort(key=lambda item: (item["source"], str(item.get("ref") or "")))
    return invalid


class _RedactionWalker:
    """单篇文档的只读遍历：收集需要替换为 ``MASK`` 的实例位置（token 序列）。

    ``_recorded`` 是祖先 / 后代不重叠的位置集合；``_expanded`` 记录已展开
    的 (引用目标, 实例位置) 对，是递归引用的终止保护。两者都是单文档状态，
    每篇文档使用独立的 walker；分支校验器缓存可跨文档共享（无状态）。
    """

    def __init__(self, schema: Any, validator: Any, branch_cache: dict):
        self._schema = schema
        self._validator = validator
        self._branch_cache = branch_cache
        self._recorded: set[tuple] = set()
        self._expanded: set[tuple[str, str]] = set()

    def collect(self, document: Any) -> set[tuple]:
        """在原始文档上只读遍历，返回需替换的实例位置集合。"""
        self._walk(self._schema, document, (), ())
        return self._recorded

    # ------------------------------------------------------------------
    # 位置记录（维持祖先 / 后代不重叠）
    # ------------------------------------------------------------------

    def _record(self, tokens: tuple) -> None:
        # 祖先位置已遮蔽整棵子树：更深层位置不再重复记录
        if any(tokens[:end] in self._recorded for end in range(1, len(tokens))):
            return
        # 新位置整体遮蔽子树：丢弃此前记录的更深层位置，以祖先为准
        self._recorded = {
            kept for kept in self._recorded if kept[:len(tokens)] != tokens
        }
        self._recorded.add(tokens)

    # ------------------------------------------------------------------
    # 遍历
    # ------------------------------------------------------------------

    def _walk(self, node: Any, value: Any, inst_tokens: tuple,
              schema_tokens: tuple) -> None:
        if node is True or node is False or not isinstance(node, dict):
            return
        if node.get("x-sensitive") is True:
            self._record(inst_tokens)
            return
        ref = node.get("$ref")
        if isinstance(ref, str):
            # draft-07 语义：$ref 接管整个节点，兄弟关键字不再生效
            self._walk_ref(ref, value, inst_tokens)
            return
        if isinstance(value, dict):
            self._walk_object(node, value, inst_tokens, schema_tokens)
        elif isinstance(value, list):
            self._walk_array(node, value, inst_tokens, schema_tokens)
        self._walk_compositions(node, value, inst_tokens, schema_tokens)

    def _walk_ref(self, ref: str, value: Any, inst_tokens: tuple) -> None:
        target_pointer, invalid = refgraph._classify_ref(self._schema, ref)
        if invalid is not None:
            # 无法解析的引用由 find_invalid_refs 统一上报（调用方返回 422）
            return
        key = (target_pointer, render_instance_pointer(inst_tokens))
        if key in self._expanded:
            # 同一实例位置的递归引用：停止重复展开
            return
        self._expanded.add(key)
        segments = refgraph._pointer_segments(target_pointer)
        node = self._schema
        for token in segments:
            node = node[int(token)] if isinstance(node, list) else node[token]
        self._walk(node, value, inst_tokens, tuple(segments))

    def _walk_object(self, node: dict, value: dict, inst_tokens: tuple,
                     schema_tokens: tuple) -> None:
        props = node.get("properties")
        if isinstance(props, dict):
            for name, sub in props.items():
                if name in value:
                    self._walk(sub, value[name],
                               inst_tokens + (name,),
                               schema_tokens + ("properties", name))

    def _walk_array(self, node: dict, value: list, inst_tokens: tuple,
                    schema_tokens: tuple) -> None:
        consumed = 0
        prefix = node.get("prefixItems")
        if isinstance(prefix, list):
            for index, sub in enumerate(prefix):
                if index < len(value):
                    self._walk(sub, value[index],
                               inst_tokens + (index,),
                               schema_tokens + ("prefixItems", str(index)))
            consumed = len(prefix)
        items = node.get("items")
        if isinstance(items, list):  # draft-04/06/07 元组写法
            for index, sub in enumerate(items):
                if index < len(value):
                    self._walk(sub, value[index],
                               inst_tokens + (index,),
                               schema_tokens + ("items", str(index)))
        elif isinstance(items, (dict, bool)):
            for index in range(consumed, len(value)):
                self._walk(items, value[index],
                           inst_tokens + (index,),
                           schema_tokens + ("items",))

    def _walk_compositions(self, node: dict, value: Any, inst_tokens: tuple,
                           schema_tokens: tuple) -> None:
        all_of = node.get("allOf")
        if isinstance(all_of, list):
            for index, branch in enumerate(all_of):
                self._walk(branch, value, inst_tokens,
                           schema_tokens + ("allOf", str(index)))
        for keyword in ("oneOf", "anyOf"):
            branches = node.get(keyword)
            if isinstance(branches, list):
                for index, branch in enumerate(branches):
                    branch_tokens = schema_tokens + (keyword, str(index))
                    if self._branch_valid(branch_tokens, value):
                        self._walk(branch, value, inst_tokens, branch_tokens)

    def _branch_valid(self, branch_tokens: tuple, value: Any) -> bool:
        """当前实例是否通过指定分支（分支内 #/... 引用解析到原始 schema）。"""
        pointer = "#" if not branch_tokens else "#/" + "/".join(
            escape_pointer(str(token)) for token in branch_tokens
        )
        validator = self._branch_cache.get(pointer)
        if validator is None:
            # 以根 schema 校验器演进出的 $ref 包装校验器：分支内的文档内
            # 引用仍解析到原始 schema，而非分支自身
            validator = self._validator.evolve(schema={"$ref": pointer})
            self._branch_cache[pointer] = validator
        return validator.is_valid(value)


def _apply_replacements(document: Any, recorded: set[tuple]) -> Any:
    """在深拷贝上把全部记录位置替换为 ``MASK``（不改动入参原文档）。"""
    if () in recorded:
        return MASK
    redacted = copy.deepcopy(document)
    for tokens in recorded:
        node = redacted
        for token in tokens[:-1]:
            node = node[token]
        node[tokens[-1]] = MASK
    return redacted


def redact_documents(schema: Any, documents: list[Any]) -> dict:
    """对一批文档执行 ``x-sensitive`` 脱敏（纯函数）。

    按输入顺序返回每篇文档的脱敏结果与替换位置实例指针；入参文档与
    schema 均不被修改。``documents`` 由调用方保证为
    1..MAX_REDACT_DOCUMENTS 个任意 JSON 值，且 schema 中不存在无法
    解析的 ``$ref``（应先经 ``find_invalid_refs`` 检查）。
    """
    validator = make_validator(schema)
    branch_cache: dict[str, Any] = {}

    results: list[dict] = []
    redacted_documents = 0
    total_replacements = 0
    for index, document in enumerate(documents):
        walker = _RedactionWalker(schema, validator, branch_cache)
        recorded = walker.collect(document)
        pointers = sorted(render_instance_pointer(tokens) for tokens in recorded)
        if pointers:
            redacted_documents += 1
        total_replacements += len(pointers)
        results.append({
            "index": index,
            "document": _apply_replacements(document, recorded),
            "redacted_paths": pointers,
        })

    return {
        "total_documents": len(documents),
        "redacted_documents": redacted_documents,
        "total_replacements": total_replacements,
        "documents": results,
    }
