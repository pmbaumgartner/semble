from __future__ import annotations

import re
from dataclasses import dataclass

_AST_NODE_SPLIT_RE = re.compile(r"[^a-z0-9]+")

_IDENTIFIER_NODE_TYPES = frozenset(
    {
        "field_identifier",
        "identifier",
        "property_identifier",
        "shorthand_property_identifier",
        "type_identifier",
    }
)
_NUMBER_NODE_TYPES = frozenset(
    {
        "decimal_integer_literal",
        "float",
        "float_literal",
        "integer",
        "integer_literal",
        "number",
        "number_literal",
    }
)
_STRING_NODE_TYPES = frozenset(
    {
        "char_literal",
        "character_literal",
        "interpreted_string_literal",
        "raw_string_literal",
        "string",
        "string_fragment",
        "string_literal",
    }
)
_BOOL_NODE_TYPES = frozenset({"bool", "boolean", "boolean_literal", "false", "false_literal", "true", "true_literal"})
_NULL_NODE_TYPES = frozenset({"nil", "null", "null_literal", "nullptr"})
_NONE_NODE_TYPES = frozenset({"none", "none_literal"})
_BEHAVIOR_NODE_TOKENS = frozenset(
    {
        "await",
        "binary",
        "call",
        "catch",
        "closure",
        "comprehension",
        "except",
        "for",
        "if",
        "lambda",
        "loop",
        "match",
        "raise",
        "return",
        "switch",
        "throw",
        "try",
        "unary",
        "update",
        "while",
        "yield",
    }
)
_DATA_SHAPE_NODE_TYPES = frozenset(
    {
        "array",
        "array_expression",
        "composite_literal",
        "dict",
        "dictionary",
        "hash",
        "hash_literal",
        "keyed_element",
        "list",
        "literal_element",
        "literal_value",
        "map_literal",
        "object",
        "object_literal",
        "pair",
        "set",
        "struct_expression",
        "tuple",
        "tuple_expression",
    }
)
_STATIC_BINDING_NODE_TYPES = frozenset(
    {
        "assignment",
        "assignment_statement",
        "const_declaration",
        "lexical_declaration",
        "short_var_declaration",
        "variable_declaration",
        "variable_declarator",
    }
)
_SCAFFOLDING_NODE_TYPES = frozenset(
    {
        "annotation",
        "annotation_argument_list",
        "attribute",
        "attribute_group",
        "attribute_item",
        "attribute_list",
        "decorator",
        "export_clause",
        "export_specifier",
        "export_statement",
        "import_declaration",
        "import_from_statement",
        "import_spec",
        "import_spec_list",
        "import_statement",
        "inner_attribute_item",
        "marker_annotation",
        "namespace_declaration",
        "namespace_definition",
        "namespace_use_clause",
        "namespace_use_declaration",
        "package_clause",
        "package_declaration",
        "php_tag",
        "scoped_use_list",
        "use_declaration",
        "use_list",
        "using_declaration",
        "using_directive",
    }
)
_STRIPPABLE_SCAFFOLDING_NODE_TYPES = frozenset(
    {
        "annotation",
        "annotation_argument_list",
        "attribute_group",
        "attribute_item",
        "attribute_list",
        "decorator",
        "import_declaration",
        "import_from_statement",
        "import_statement",
        "inner_attribute_item",
        "marker_annotation",
        "namespace_use_clause",
        "namespace_use_declaration",
        "package_clause",
        "package_declaration",
        "php_tag",
        "scoped_use_list",
        "use_declaration",
        "use_list",
        "using_declaration",
        "using_directive",
    }
)
_TRANSPARENT_SCAFFOLDING_NODE_TYPES = frozenset(
    {
        "export_statement",
        "namespace_declaration",
        "namespace_definition",
    }
)
_NON_SUBSTANTIVE_NODE_TYPES = frozenset(
    {
        "annotation_type_body",
        "argument_list",
        "array_type",
        "block",
        "body",
        "class_body",
        "compound_statement",
        "declaration_list",
        "dimensions",
        "enum_body",
        "formal_parameters",
        "generic_type",
        "interface_body",
        "modifier",
        "modifiers",
        "module",
        "parameter_list",
        "parameters",
        "predefined_type",
        "primitive_type",
        "program",
        "qualified_type",
        "script",
        "source_file",
        "token_tree",
        "translation_unit",
        "type",
        "type_annotation",
        "visibility_modifier",
    }
)


@dataclass(frozen=True, slots=True)
class AstStats:
    """Parser-derived counts used to decide duplicate eligibility."""

    code_bearing: int = 0
    behavioral: int = 0
    data_shape: int = 0
    static_binding: int = 0
    scaffolding: int = 0
    substantive: int = 0

    def __add__(self, other: AstStats) -> AstStats:
        """Combine two AST stat snapshots."""
        return AstStats(
            code_bearing=self.code_bearing + other.code_bearing,
            behavioral=self.behavioral + other.behavioral,
            data_shape=self.data_shape + other.data_shape,
            static_binding=self.static_binding + other.static_binding,
            scaffolding=self.scaffolding + other.scaffolding,
            substantive=self.substantive + other.substantive,
        )


def _stats_for_node(node_type: str) -> AstStats:
    return AstStats(
        code_bearing=1,
        behavioral=1 if _is_behavior_ast_node(node_type) else 0,
        data_shape=1 if _is_data_shape_ast_node(node_type) else 0,
        static_binding=1 if _is_static_binding_ast_node(node_type) else 0,
        scaffolding=1 if _is_scaffolding_ast_node(node_type) else 0,
        substantive=1 if _is_substantive_ast_node(node_type) else 0,
    )


def _is_ignored_ast_subtree(node_type: str) -> bool:
    lower = node_type.lower()
    return lower == "comment" or lower.endswith("_comment") or _normalize_ast_label(lower) == "STRING"


def _is_behavior_ast_node(node_type: str) -> bool:
    return any(token in _BEHAVIOR_NODE_TOKENS for token in _ast_node_tokens(node_type))


def _is_data_shape_ast_node(node_type: str) -> bool:
    lower = node_type.lower()
    return lower in _DATA_SHAPE_NODE_TYPES or (
        lower.endswith("_literal") and "char" not in lower and "string" not in lower
    )


def _is_static_binding_ast_node(node_type: str) -> bool:
    return node_type.lower() in _STATIC_BINDING_NODE_TYPES


def _is_scaffolding_ast_node(node_type: str) -> bool:
    lower = node_type.lower()
    tokens = _ast_node_tokens(lower)
    if lower in _SCAFFOLDING_NODE_TYPES:
        return True
    if any(token in {"import", "namespace", "package"} for token in tokens):
        return True
    if "annotation" in tokens and "declaration" not in tokens:
        return True
    if "attribute" in tokens and "declaration" not in tokens:
        return True
    return False


def _is_strippable_scaffolding_node(node_type: str) -> bool:
    return node_type.lower() in _STRIPPABLE_SCAFFOLDING_NODE_TYPES


def _is_transparent_scaffolding_node(node_type: str) -> bool:
    return node_type.lower() in _TRANSPARENT_SCAFFOLDING_NODE_TYPES


def _is_substantive_ast_node(node_type: str) -> bool:
    lower = node_type.lower()
    if _is_scaffolding_ast_node(lower):
        return False
    if _is_name_ast_node(lower):
        return False
    return lower not in _NON_SUBSTANTIVE_NODE_TYPES


def _is_name_ast_node(node_type: str) -> bool:
    lower = node_type.lower()
    return (
        lower in _IDENTIFIER_NODE_TYPES
        or lower.endswith("identifier")
        or lower in {"dotted_name", "name", "namespace_name", "qualified_name", "scoped_identifier"}
    )


def _ast_node_tokens(node_type: str) -> list[str]:
    return [token for token in _AST_NODE_SPLIT_RE.split(node_type.lower()) if token]


def _normalize_ast_label(node_type: str) -> str:
    lower = node_type.lower()
    if lower in _IDENTIFIER_NODE_TYPES or lower.endswith("identifier"):
        return "IDENT"
    if lower in _NUMBER_NODE_TYPES or "number" in lower or lower.endswith("_integer_literal"):
        return "NUMBER"
    if lower in _STRING_NODE_TYPES or "string" in lower:
        return "STRING"
    if lower in _BOOL_NODE_TYPES:
        return "BOOL"
    if lower in _NULL_NODE_TYPES:
        return "NULL"
    if lower in _NONE_NODE_TYPES:
        return "NONE"
    return node_type
