
"""Minimal XJSON Control Flow Engine (XCFE) for trainers."""
from typing import Any, Dict

def _get_from_ctx(ref: str, ctx: Dict[str, Any]) -> Any:
    """Resolve "$a.b.c" style refs from a nested context dict."""
    if not isinstance(ref, str) or not ref.startswith("$"):
        return ref
    path = ref[1:].split(".")
    cur: Any = ctx
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

def _cmp(left: Any, op: str, right: Any) -> bool:
    if op == "==": return left == right
    if op == "!=": return left != right
    if op == ">":  return left > right
    if op == "<":  return left < right
    if op == ">=": return left >= right
    if op == "<=": return left <= right
    if op == "contains": return str(right) in str(left)
    if op == "startswith": return str(left).startswith(str(right))
    if op == "endswith": return str(left).endswith(str(right))
    if op == "in": return left in right
    if op == "not_in": return left not in right
    if op == "isnull": return left is None
    if op == "notnull": return left is not None
    return False

def eval_if_block(block: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    """Return "then" or "else" depending on which branch should fire."""
    node = block.get("@if") or block
    cond = node.get("cond") or {}
    left = _get_from_ctx(cond.get("left"), ctx)
    right = _get_from_ctx(cond.get("right"), ctx)
    op = cond.get("op", "==")
    result = _cmp(left, op, right)
    return "then" if result else "else"
