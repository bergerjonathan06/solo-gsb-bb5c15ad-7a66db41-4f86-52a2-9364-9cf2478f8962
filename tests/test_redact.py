"""敏感字段脱敏引擎单元测试。

覆盖：对象属性 / 数组元素遍历、x-sensitive 命中与整体替换、文档内 $ref
解析（含递归终止）、oneOf/anyOf 分支选择、allOf 全部分支、祖先 / 后代
位置去重、无法解析引用扫描、原文档不被修改、输入顺序保持。
"""
import copy

from app import redact


# ---------------------------------------------------------------------------
# 对象属性与数组元素
# ---------------------------------------------------------------------------


def test_masks_marked_object_properties():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "password": {"type": "string", "x-sensitive": True},
        },
    }
    result = redact.redact_documents(schema, [
        {"name": "alice", "password": "s3cret"},
        {"name": "bob"},
    ])
    assert result["total_documents"] == 2
    assert result["redacted_documents"] == 1
    assert result["total_replacements"] == 1

    first = result["documents"][0]
    assert first["index"] == 0
    assert first["document"] == {"name": "alice", "password": "***"}
    assert first["redacted_paths"] == ["#/password"]

    # 属性缺失：无替换、无指针
    second = result["documents"][1]
    assert second["document"] == {"name": "bob"}
    assert second["redacted_paths"] == []


def test_masks_nested_properties_and_reports_sorted_pointers():
    schema = {
        "properties": {
            "user": {
                "properties": {
                    "token": {"x-sensitive": True},
                    "profile": {"properties": {"ssn": {"x-sensitive": True}}},
                }
            }
        }
    }
    doc = {"user": {"token": "t", "profile": {"ssn": "1", "keep": 2}}}
    result = redact.redact_documents(schema, [doc])
    entry = result["documents"][0]
    assert entry["document"] == {
        "user": {"token": "***", "profile": {"ssn": "***", "keep": 2}}
    }
    assert entry["redacted_paths"] == ["#/user/profile/ssn", "#/user/token"]


def test_masks_regardless_of_instance_type():
    # 类型不匹配也照掩：脱敏是兜底操作，不依赖实例通过校验
    schema = {"properties": {"secret": {"type": "string", "x-sensitive": True}}}
    result = redact.redact_documents(schema, [{"secret": 123}, {"secret": None}])
    assert result["documents"][0]["document"] == {"secret": "***"}
    assert result["documents"][1]["document"] == {"secret": "***"}


def test_only_literal_true_triggers_masking():
    schema = {"properties": {
        "a": {"x-sensitive": "true"},
        "b": {"x-sensitive": 1},
        "c": {"x-sensitive": False},
    }}
    result = redact.redact_documents(schema, [{"a": 1, "b": 2, "c": 3}])
    assert result["documents"][0]["document"] == {"a": 1, "b": 2, "c": 3}
    assert result["total_replacements"] == 0


def test_root_sensitive_masks_whole_document():
    schema = {"x-sensitive": True}
    result = redact.redact_documents(schema, [{"a": 1}, "text", None, [1]])
    assert result["redacted_documents"] == 4
    for entry in result["documents"]:
        assert entry["document"] == "***"
        assert entry["redacted_paths"] == ["#"]


def test_array_items_homogeneous():
    schema = {"items": {"x-sensitive": True}}
    result = redact.redact_documents(schema, [["a", "b"], []])
    assert result["documents"][0]["document"] == ["***", "***"]
    assert result["documents"][0]["redacted_paths"] == ["#/0", "#/1"]
    assert result["documents"][1]["redacted_paths"] == []


def test_array_prefix_items():
    schema = {"prefixItems": [{"x-sensitive": True}, {"type": "string"}]}
    result = redact.redact_documents(schema, [["s", "keep", "extra"]])
    assert result["documents"][0]["document"] == ["***", "keep", "extra"]
    assert result["documents"][0]["redacted_paths"] == ["#/0"]


def test_array_prefix_items_then_items_dict():
    # 2020-12：items 只作用于 prefixItems 之后的位置
    schema = {
        "prefixItems": [{"type": "string"}],
        "items": {"x-sensitive": True},
    }
    result = redact.redact_documents(schema, [["keep", "s1", "s2"]])
    assert result["documents"][0]["document"] == ["keep", "***", "***"]
    assert result["documents"][0]["redacted_paths"] == ["#/1", "#/2"]


def test_array_legacy_tuple_items():
    # draft-04/06/07 元组写法：items 为列表
    schema = {"items": [{"x-sensitive": True}, {"x-sensitive": True}]}
    result = redact.redact_documents(schema, [["a", "b", "c"]])
    assert result["documents"][0]["document"] == ["***", "***", "c"]
    assert result["documents"][0]["redacted_paths"] == ["#/0", "#/1"]


# ---------------------------------------------------------------------------
# 组合关键字
# ---------------------------------------------------------------------------


def test_all_of_traverses_every_branch():
    schema = {
        "allOf": [
            {"properties": {"a": {"x-sensitive": True}}},
            {"properties": {"b": {"x-sensitive": True}}},
        ]
    }
    result = redact.redact_documents(schema, [{"a": 1, "b": 2}])
    assert result["documents"][0]["document"] == {"a": "***", "b": "***"}
    assert result["documents"][0]["redacted_paths"] == ["#/a", "#/b"]


def test_one_of_only_valid_branch_traversed():
    schema = {
        "oneOf": [
            {"required": ["card"], "properties": {"card": {"x-sensitive": True}}},
            {"required": ["paypal"], "properties": {"paypal": {"x-sensitive": True}}},
        ]
    }
    result = redact.redact_documents(schema, [
        {"card": "4111"},          # 仅分支 0 校验通过
        {"paypal": "me@x.com"},    # 仅分支 1 校验通过
        {"other": 1},              # 两个分支都不通过 -> 不脱敏
    ])
    assert result["documents"][0]["document"] == {"card": "***"}
    assert result["documents"][0]["redacted_paths"] == ["#/card"]
    assert result["documents"][1]["document"] == {"paypal": "***"}
    assert result["documents"][1]["redacted_paths"] == ["#/paypal"]
    assert result["documents"][2]["document"] == {"other": 1}
    assert result["documents"][2]["redacted_paths"] == []


def test_any_of_traverses_all_valid_branches():
    schema = {
        "anyOf": [
            {"properties": {"a": {"x-sensitive": True}}},
            {"properties": {"b": {"x-sensitive": True}}},
        ]
    }
    # 两个分支都校验通过（均无 required），a 与 b 都被脱敏
    result = redact.redact_documents(schema, [{"a": 1, "b": 2}])
    assert result["documents"][0]["document"] == {"a": "***", "b": "***"}
    assert result["documents"][0]["redacted_paths"] == ["#/a", "#/b"]


def test_branch_selection_sees_original_values():
    # 先经 properties 遮蔽了 kind，oneOf 分支选择仍基于原始值判定
    schema = {
        "properties": {"kind": {"x-sensitive": True}},
        "oneOf": [
            {"required": ["kind"],
             "properties": {"kind": {"const": "a"},
                            "secret": {"x-sensitive": True}}},
        ],
    }
    result = redact.redact_documents(schema, [{"kind": "a", "secret": "s"}])
    entry = result["documents"][0]
    assert entry["document"] == {"kind": "***", "secret": "***"}
    assert entry["redacted_paths"] == ["#/kind", "#/secret"]


def test_ancestor_mask_subsumes_descendant():
    # allOf 另一分支到达的更深层位置被祖先遮蔽，只报告祖先指针
    schema = {
        "properties": {"a": {"x-sensitive": True}},
        "allOf": [{"properties": {"a": {"properties": {"b": {"x-sensitive": True}}}}}],
    }
    result = redact.redact_documents(schema, [{"a": {"b": "deep"}}])
    entry = result["documents"][0]
    assert entry["document"] == {"a": "***"}
    assert entry["redacted_paths"] == ["#/a"]
    assert result["total_replacements"] == 1


# ---------------------------------------------------------------------------
# 文档内 $ref
# ---------------------------------------------------------------------------


def test_ref_target_sensitive_marker_applies():
    schema = {
        "properties": {"pw": {"$ref": "#/$defs/Secret"}},
        "$defs": {"Secret": {"type": "string", "x-sensitive": True}},
    }
    result = redact.redact_documents(schema, [{"pw": "s3cret"}])
    assert result["documents"][0]["document"] == {"pw": "***"}
    assert result["documents"][0]["redacted_paths"] == ["#/pw"]


def test_recursive_ref_terminates_and_masks_all_levels():
    schema = {
        "$ref": "#/$defs/Node",
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "secret": {"type": "string", "x-sensitive": True},
                    "child": {"$ref": "#/$defs/Node"},
                },
            }
        },
    }
    doc = {"secret": "a", "child": {"secret": "b", "child": {"secret": "c"}}}
    result = redact.redact_documents(schema, [doc])
    entry = result["documents"][0]
    assert entry["document"] == {
        "secret": "***",
        "child": {"secret": "***", "child": {"secret": "***"}},
    }
    assert entry["redacted_paths"] == [
        "#/child/child/secret", "#/child/secret", "#/secret",
    ]


def test_pure_ref_cycle_terminates():
    # A <-> B 互引：同一实例位置重复展开被守卫终止
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/A"}},
        "$defs": {
            "A": {"$ref": "#/$defs/B"},
            "B": {"$ref": "#/$defs/A"},
        },
    }
    result = redact.redact_documents(schema, [{"x": 1}])
    assert result["documents"][0]["document"] == {"x": 1}
    assert result["documents"][0]["redacted_paths"] == []


def test_self_root_ref_through_property_terminates():
    # properties/again 引用文档根：随实例深度展开，实例耗尽即终止
    schema = {
        "type": "object",
        "properties": {
            "v": {"x-sensitive": True},
            "again": {"$ref": "#"},
        },
    }
    doc = {"v": 1, "again": {"v": 2, "again": {"v": 3}}}
    result = redact.redact_documents(schema, [doc])
    entry = result["documents"][0]
    assert entry["document"] == {
        "v": "***",
        "again": {"v": "***", "again": {"v": "***"}},
    }
    assert entry["redacted_paths"] == ["#/again/again/v", "#/again/v", "#/v"]


def test_ref_inside_one_of_branch_resolves_against_root():
    # oneOf 分支内的 #/... 引用仍解析到原始 schema 的 $defs
    schema = {
        "$defs": {"Secret": {"x-sensitive": True}},
        "oneOf": [
            {"required": ["a"], "properties": {"a": {"$ref": "#/$defs/Secret"}}},
            {"required": ["b"]},
        ],
    }
    result = redact.redact_documents(schema, [{"a": "s"}, {"b": 1}])
    assert result["documents"][0]["document"] == {"a": "***"}
    assert result["documents"][0]["redacted_paths"] == ["#/a"]
    assert result["documents"][1]["redacted_paths"] == []


# ---------------------------------------------------------------------------
# 无法解析的引用
# ---------------------------------------------------------------------------


def test_find_invalid_refs_reports_schema_pointer():
    schema = {
        "properties": {
            "a": {"$ref": "#/$defs/Missing"},
            "b": {"$ref": "https://example.com/x.json#/$defs/Y"},
            "c": {"$ref": "#anchor"},
            "d": {"$ref": "#/properties/~2"},
        },
        "$defs": {"ok": {"type": "string"}},
    }
    invalid = redact.find_invalid_refs(schema)
    by_source = {item["source"]: item for item in invalid}
    assert by_source["#/properties/a"]["reason"] == "target_not_found"
    assert by_source["#/properties/b"]["reason"] == "external_reference"
    assert by_source["#/properties/c"]["reason"] == "name_fragment_unsupported"
    assert by_source["#/properties/d"]["reason"] == "malformed_pointer"


def test_find_invalid_refs_clean_schema():
    schema = {
        "properties": {"x": {"$ref": "#/$defs/A"}},
        "$defs": {"A": {"type": "string"}},
    }
    assert redact.find_invalid_refs(schema) == []


# ---------------------------------------------------------------------------
# 只读性与确定性
# ---------------------------------------------------------------------------


def test_input_documents_are_not_modified():
    schema = {"properties": {"pw": {"x-sensitive": True}}}
    docs = [{"pw": "secret", "keep": [1, {"nested": "v"}]}]
    snapshot = copy.deepcopy(docs)
    redact.redact_documents(schema, docs)
    assert docs == snapshot


def test_schema_is_not_modified():
    schema = {
        "$ref": "#/$defs/Node",
        "$defs": {"Node": {"properties": {"s": {"x-sensitive": True}}}},
    }
    snapshot = copy.deepcopy(schema)
    redact.redact_documents(schema, [{"s": "x"}])
    assert schema == snapshot


def test_output_order_matches_input_and_is_deterministic():
    schema = {"properties": {"pw": {"x-sensitive": True}}}
    docs = [{"pw": i} for i in range(5)] + [{"nopw": 1}]
    first = redact.redact_documents(schema, docs)
    second = redact.redact_documents(schema, copy.deepcopy(docs))
    assert first == second
    assert [d["index"] for d in first["documents"]] == list(range(6))
    assert first["documents"][4]["document"] == {"pw": "***"}
    assert first["documents"][5]["document"] == {"nopw": 1}
    assert first["redacted_documents"] == 5
    assert first["total_replacements"] == 5
