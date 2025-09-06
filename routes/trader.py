import logging
import math
import re
import ast
import json
from typing import Dict, List, Any, Tuple, Optional
import keyword

from flask import request, jsonify
from routes import app

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FAIL_LOGGER = logging.getLogger("failed_tests")
FAIL_LOGGER.setLevel(logging.ERROR)

def _log_failed_test(name: str, formula: str, variables: Dict[str, Any], ttype: str,
                     phase: str, kind: str, msg: str) -> None:
    """Emit exactly one JSON line per failed testcase with only input + compact error."""
    entry = {
        "event": "test_failed",
        "name": name,
        "type": ttype,
        "formula": formula,
        "variables": variables,
        "error": {"phase": phase, "kind": kind, "msg": (msg or "")[:200]}
    }
    FAIL_LOGGER.error(json.dumps(entry, ensure_ascii=False))

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DEFAULT_RESULT = 0.0   # fallback value for a failing testcase
ROUND_DP = 4
SUM_NEST_LIMIT = 50

# ----------------------------------------------------------------------
# Safe evaluation (restricted AST)
# ----------------------------------------------------------------------
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num, ast.Constant,
    ast.Name, ast.Load,
    ast.Call, ast.Attribute,
    ast.Tuple, ast.List,
)
_ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)

def _fix_accidental_variable_calls(expr: str, env: Dict[str, Any]) -> str:
    """
    Replace accidental 'var(' with 'var*(' when 'var' exists in env and is NOT callable.
    This preserves real functions like max/min/math.log, since those are callable.
    """
    # Sort by descending length to avoid partial overlaps (e.g., sigma_10 before sigma_1)
    names = sorted([k for k in env.keys() if isinstance(k, str)], key=len, reverse=True)

    for name in names:
        val = env.get(name)
        # skip callables/modules and dunders
        if callable(val): 
            continue
        if name.startswith("__"):
            continue
        # only patch plain identifiers (avoid 'math.log', which has a dot)
        if "." in name:
            continue

        # Replace occurrences of \bname\s*( with name*(
        # Use a negative lookbehind to avoid replacing attribute calls like math.log(
        pattern = rf'(?<!\.)\b{re.escape(name)}\s*\('
        expr = re.sub(pattern, f'{name}*(', expr)
    return expr


def _safe_eval(expr: str, env: Dict[str, Any]) -> float:
    """Safely evaluate arithmetic expression using a restricted AST."""
    def _check(node: ast.AST):
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BIN_OPS):
                raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
            _check(node.left); _check(node.right); return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_OPS):
                raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")
            _check(node.operand); return
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in env:
                    raise ValueError(f"Call to unknown function: {func.id}")
            elif isinstance(func, ast.Attribute):
                if not (isinstance(func.value, ast.Name) and func.value.id == "math"):
                    raise ValueError("Only math.<func> attribute calls are allowed")
            else:
                raise ValueError("Unsupported callable")
            for a in node.args: _check(a)
            for kw in (node.keywords or []): _check(kw.value)
            return
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"):
                raise ValueError("Only math.<attr> is allowed")
            return
        if isinstance(node, (ast.Name, ast.Constant)):
            return
        if isinstance(node, (ast.Tuple, ast.List)):
            for ch in ast.iter_child_nodes(node): _check(ch)
            return
        if isinstance(node, ast.Expression):
            _check(node.body); return
        raise ValueError(f"Disallowed expression node: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    _check(tree)
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env)

# ----------------------------------------------------------------------
# LaTeX preprocessing helpers
# ----------------------------------------------------------------------
_GREEK = {
    r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma", r"\delta": "delta",
    r"\epsilon": "epsilon", r"\zeta": "zeta", r"\eta": "eta", r"\theta": "theta",
    r"\iota": "iota", r"\kappa": "kappa", r"\lambda": "lambda", r"\mu": "mu",
    r"\nu": "nu", r"\xi": "xi", r"\pi": "pi", r"\rho": "rho", r"\sigma": "sigma",
    r"\tau": "tau", r"\upsilon": "upsilon", r"\phi": "phi", r"\chi": "chi",
    r"\psi": "psi", r"\omega": "omega",
    r"\Gamma": "Gamma", r"\Delta": "Delta", r"\Theta": "Theta", r"\Lambda": "Lambda",
    r"\Xi": "Xi", r"\Pi": "Pi", r"\Sigma": "Sigma", r"\Upsilon": "Upsilon",
    r"\Phi": "Phi", r"\Psi": "Psi", r"\Omega": "Omega",
}

_SPACING_TOKENS = [
    r"\left", r"\right", r"\Big", r"\big", r"\Bigg", r"\bigg",
    r"\!", r"\,", r"\;", r"\:", r"\ ", r"\quad", r"\qquad"
]

def _strip_math_delims(s: str) -> str:
    s = s.strip()
    if s.startswith("$$") and s.endswith("$$"): return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$"):  return s[1:-1].strip()
    return s

def _rhs_of_equals(s: str) -> str:
    return s.split("=", 1)[1].strip() if "=" in s else s

def _replace_text_like(s: str) -> str:
    # \text{Trade Amount} -> TradeAmount ; \mathrm{NPV} -> NPV
    def joiner(m): return re.sub(r"\s+", "", m.group(1))
    s = re.sub(r"\\text\{([^}]*)\}", joiner, s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", joiner, s)
    return s

def _replace_greek(s: str) -> str:
    for k, v in _GREEK.items(): s = s.replace(k, v)
    return s

def _resolve_t_subscripts(s: str, default_t: int = 1) -> str:
    """Turn _{t-k} -> _<default_t-k>, and _{t} -> _<default_t>."""
    def repl_minus(m):
        k = int(m.group(1))
        return f"_{max(default_t - k, 0)}"
    s = re.sub(r"_\{t-([0-9]+)\}", repl_minus, s)
    s = re.sub(r"_\{t\}", f"_{default_t}", s)
    return s

def _normalize_indices(s: str) -> str:
    # E[R_m] -> E_R_m ; A[B] -> A_B ; X_{a_b} -> X_a_b
    s = re.sub(r"([A-Za-z]+)\[([^\]]+)\]", lambda m: f"{m.group(1)}_{m.group(2)}", s)
    s = re.sub(r"_\{([^}]+)\}", lambda m: "_" + m.group(1), s)
    s = re.sub(r"\s*_\s*", "_", s)
    return s

def _normalize_basic_ops(s: str) -> str:
    # Multiplication
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\cdotp", "*")
    # Max/Min (and generic \operatorname{Foo} -> Foo)
    s = s.replace(r"\operatorname{Max}", "max").replace(r"\operatorname{Min}", "min")
    s = s.replace(r"\max", "max").replace(r"\min", "min")
    s = re.sub(r"\\operatorname\{([^}]+)\}", r"\1", s)
    # Remove size/spacing macros
    for tok in _SPACING_TOKENS:
        s = s.replace(tok, "")
    return s

def _find_braced(s: str, i: int) -> Tuple[str, int]:
    """Given s and index i at '{', return (content, end_index_of_closing_brace)."""
    assert s[i] == "{"
    depth = 0; j = i
    while j < len(s):
        if s[j] == "{": depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0: return s[i+1:j], j
        j += 1
    raise ValueError("Unbalanced braces in LaTeX")

def _transform_frac(s: str) -> str:
    """Replace all \frac{A}{B} with ((A)/(B)), handling nesting."""
    out = []; i = 0
    while i < len(s):
        if s.startswith(r"\frac", i):
            i += len(r"\frac")
            if i >= len(s) or s[i] != "{": raise ValueError(r"Expected '{' after \frac")
            num, j = _find_braced(s, i); i = j + 1
            if i >= len(s) or s[i] != "{": raise ValueError(r"Expected '{' for denominator in \frac")
            den, j = _find_braced(s, i); i = j + 1
            out.append(f"(({_transform_frac(num)})/({_transform_frac(den)}))")
        else:
            out.append(s[i]); i += 1
    return "".join(out)

def _transform_sqrt(s: str) -> str:
    """Replace \sqrt{X} with ((X)**0.5)."""
    out = []; i = 0
    while i < len(s):
        if s.startswith(r"\sqrt", i):
            i += len(r"\sqrt")
            if i >= len(s) or s[i] != "{": raise ValueError(r"Expected '{' after \sqrt")
            inner, j = _find_braced(s, i)
            i = j + 1
            out.append(f"(({inner})**0.5)")
        else:
            out.append(s[i]); i += 1
    return "".join(out)

def _transform_power_e(s: str) -> str:
    """Handle e^{...} -> (e**(...)). Other '^' converted later."""
    out = []; i = 0
    while i < len(s):
        if s.startswith("e^{", i):
            i += 2
            if i >= len(s) or s[i] != "{": raise ValueError("Malformed e^{...}")
            inner, j = _find_braced(s, i); i = j + 1
            out.append(f"(e**({inner}))")
        else:
            out.append(s[i]); i += 1
    return "".join(out)

def _normalize_funcs_to_python(s: str) -> str:
    # \log(x)/\ln(x) -> math.log(x); \exp(x) -> math.exp(x)
    s = s.replace(r"\log", "log").replace(r"\exp", "exp").replace(r"\ln", "log")
    s = re.sub(r"(?<![A-Za-z0-9_])log\s*\(", "math.log(", s)
    s = re.sub(r"(?<![A-Za-z0-9_])exp\s*\(", "math.exp(", s)
    return s

def _clean_ident_piece(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "", x)

def _find_paren_group(s: str, i: int) -> Tuple[str, int]:
    """Given s[i]=='(', return (content, end_index)."""
    assert s[i] == "("
    depth = 0; j = i
    while j < len(s):
        if s[j] == "(": depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0: return s[i+1:j], j
        j += 1
    raise ValueError("Unbalanced parentheses")

def _split_top_level_commas(s: str) -> List[str]:
    parts, depth, buf = [], 0, []
    for ch in s:
        if ch in "([{": depth += 1
        elif ch in ")]}": depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return parts

def _normalize_symbolic_wrappers(s: str) -> str:
    """
    Convert Cov(X,Y) -> Cov_X_Y and Var(X) -> Var_X so they map to JSON keys
    like Cov_DeltaS_DeltaF, Var_r_i, Var_DeltaF.
    """
    out = []; i = 0
    while i < len(s):
        if s.startswith("Cov(", i) or s.startswith("Var(", i):
            name = "Cov" if s.startswith("Cov(", i) else "Var"
            i0 = i + len(name)
            content, j = _find_paren_group(s, i0)
            i = j + 1
            if name == "Cov":
                args = _split_top_level_commas(content)
                if len(args) != 2:
                    inner = _clean_ident_piece(content)
                    out.append(f"{name}_{inner}")
                else:
                    a = _clean_ident_piece(args[0]); b = _clean_ident_piece(args[1])
                    out.append(f"{name}_{a}_{b}")
            else:
                inner = _clean_ident_piece(content)
                out.append(f"{name}_{inner}")
        else:
            out.append(s[i]); i += 1
    return "".join(out)

def _insert_implicit_mult(s: str, aggressive: bool = False) -> str:
    """
    Insert '*' for implicit multiplication while preserving function calls.
    Uses ASCII placeholders and guarantees restoration.
    """
    # ASCII placeholders for calls we must not split
    protos = {
        "math.log(": "<MATH_LOG>(",
        "math.exp(": "<MATH_EXP>(",
        "max(":      "<MAX>(",
        "min(":      "<MIN>(",
    }
    for k, v in protos.items():
        s = s.replace(k, v)

    # 1) number immediately followed by identifier: 2x -> 2*x
    s = re.sub(r'(\d)\s*([A-Za-z_])', r'\1*\2', s)

    # 2) identifier or ) or ] followed by '(' : A( -> A*(
    s = re.sub(r'([0-9A-Za-z_)\]])\s*\(', r'\1*(', s)

    # 3) ')' followed by identifier: ')x' -> ')*x'
    s = re.sub(r'\)\s*([A-Za-z_])', r')*\1', s)

    # 4) identifier whitespace identifier: 'x y' -> 'x*y'
    s = re.sub(r'([A-Za-z_][A-Za-z_0-9]*)\s+([A-Za-z_])', r'\1*\2', s)

    if aggressive:
        # Split very short glued tokens like 'bp' -> 'b*p', but never split long, real names.
        def split_short_token(m: re.Match) -> str:
            tok = m.group(0)
            if "_" in tok or len(tok) > 4:
                return tok
            return "*".join(list(tok))
        s = re.sub(r'\b[A-Za-z]{2,4}\b', split_short_token, s)

    # restore function calls
    for k, v in protos.items():
        s = s.replace(v, k)

    return s

def _strip_redundant_braces(s: str) -> str:
    return s

def _caret_to_pow(s: str) -> str:
    """
    Robustly convert a^{b} or (a)^{b} or (a)^b into (a**(b)).
    Also handles [...] as a grouped base. Any remaining '^' fall back to '**'.
    """
    def find_matching_open(src: str, pos_close: int, open_ch: str, close_ch: str) -> int:
        depth = 0
        j = pos_close
        while j >= 0:
            ch = src[j]
            if ch == close_ch:
                depth += 1
            elif ch == open_ch:
                depth -= 1
                if depth == 0:
                    return j
            j -= 1
        return -1  # not found

    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] != '^':
            out.append(s[i])
            i += 1
            continue

        # ----- Determine the base -----
        # Look one char left of '^'
        if not out:
            # no base; emit '^' literally (edge case)
            out.append('^')
            i += 1
            continue

        # We have already accumulated left-side chars in `out`;
        # inspect the last char of `out`.
        left_last = out[-1]

        # Case A: Parenthesized base (...)^X
        if left_last == ')':
            # Find matching '(' in the *already-built* out buffer.
            # Convert out -> string to search backwards.
            left_str = ''.join(out)
            pos_close = len(left_str) - 1
            pos_open = find_matching_open(left_str, pos_close, '(', ')')
            if pos_open == -1:
                # If not found, keep literal '^' (fail safe)
                out.append('^')
                i += 1
                continue
            base = left_str[pos_open:pos_close+1]  # includes parentheses
            # Remove the base we just extracted from out; we will re-insert wrapped.
            out = list(left_str[:pos_open])

        # Case B: Bracketed base [...]^X
        elif left_last == ']':
            left_str = ''.join(out)
            pos_close = len(left_str) - 1
            pos_open = find_matching_open(left_str, pos_close, '[', ']')
            if pos_open == -1:
                out.append('^')
                i += 1
                continue
            base = left_str[pos_open:pos_close+1]  # includes brackets
            out = list(left_str[:pos_open])

        # Case C: Simple token base like A_B, x1, sigma_t
        else:
            # Walk left over [A-Za-z0-9_]
            left_str = ''.join(out)
            j = len(left_str) - 1
            while j >= 0 and re.match(r'[A-Za-z0-9_]', left_str[j]):
                j -= 1
            base = left_str[j+1:]
            out = list(left_str[:j+1])
            if not base:
                # nothing found; leave '^' as-is
                out.append('^')
                i += 1
                continue

        # Move past '^'
        i += 1
        if i >= n:
            # dangling '^' at end; stitch back naive
            out.append(base + '**')
            break

        # ----- Determine the exponent -----
        if s[i] == '{':
            # brace group ^{...}
            inner, j = _find_braced(s, i)  # returns content, end_index_of_'}'
            exponent = f'({inner})'
            i = j + 1
        elif s[i] == '(':
            # parenthesis group ^(...)
            # find matching ')'
            depth = 0
            j = i
            while j < n:
                if s[j] == '(':
                    depth += 1
                elif s[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= n:
                # unmatched; treat as simple token from '(' to end
                exponent = s[i:]
                i = n
            else:
                exponent = s[i:j+1]  # keep parens
                i = j + 1
        else:
            # simple token exponent ^t or ^2
            m = re.match(r'[A-Za-z0-9_]+', s[i:])
            if m:
                tok = m.group(0)
                exponent = tok
                i += len(tok)
            else:
                # nothing usable; emit '**' and continue
                out.append(f'({base}**')
                continue

        # Emit (base**exponent) — ensure base is wrapped in () if not already
        need_wrap_base = not (base.startswith('(') and base.endswith(')')) and \
                         not (base.startswith('[') and base.endswith(']'))
        base_wrapped = f'({base})' if need_wrap_base else base
        out.append(f'({base_wrapped}**{exponent})')

    s2 = ''.join(out)
    # Fallback: convert any stray '^' to '**' (should be rare after above)
    s2 = s2.replace('^', '**')
    return s2


# ----------------------------------------------------------------------
# Constraint handling (split once, apply once)
# ----------------------------------------------------------------------
def _split_top_level_comma_once(s: str) -> Tuple[str, Optional[str]]:
    depth = 0
    for i, ch in enumerate(s):
        if ch in "([{": depth += 1
        elif ch in ")]}": depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            return s[:i].strip(), s[i+1:].strip()
    return s.strip(), None

def _latex_rhs_to_python(rhs: str) -> str:
    """Lightweight LaTeX→python for RHS of constraints (no sums)."""
    t = rhs.strip()
    t = _replace_text_like(t)
    t = _replace_greek(t)
    t = _resolve_t_subscripts(t, default_t=1)
    t = _normalize_indices(t)
    t = _normalize_basic_ops(t)
    t = _transform_frac(t)
    t = _transform_sqrt(t)
    t = _transform_power_e(t)
    t = _normalize_funcs_to_python(t)
    t = _caret_to_pow(t)
    t = t.replace("{", "(").replace("}", ")")
    t = _insert_implicit_mult(t, aggressive=False)
    t = re.sub(r"\s+", "", t)
    return t

def _apply_trailing_constraints_once(expr: str, trailing: Optional[str]) -> str:
    if not trailing:
        return expr
    parts = _split_top_level_commas(trailing)
    assigns: List[Tuple[str, str]] = []
    for p in parts:
        if "=" not in p:
            continue
        lhs, rhs = p.split("=", 1)
        lhs, rhs = lhs.strip(), rhs.strip()
        if not lhs:
            continue
        rhs_py = _latex_rhs_to_python(rhs)
        assigns.append((lhs, rhs_py))
    for lhs, rhs_py in assigns:
        expr = re.sub(rf'\b{re.escape(lhs)}\b', f'({rhs_py})', expr)
    return expr

# ----------------------------------------------------------------------
# Summation (\sum) support
# ----------------------------------------------------------------------
def _parse_sum_header(s: str, start: int) -> Optional[Tuple[int, int, str, str, str]]:
    if not s.startswith(r"\sum_", start): return None
    i = start; hdr_start = i; i += len(r"\sum_")
    # Lower: {i=1} or i=1
    if i < len(s) and s[i] == "{":
        lower, j = _find_braced(s, i); i = j + 1
    else:
        j = i
        while j < len(s) and s[j] != "^": j += 1
        lower = s[i:j].strip(); i = j
    if i >= len(s) or s[i] != "^": raise ValueError("Missing '^' in sum header")
    i += 1
    # Upper: {N} or N
    if i < len(s) and s[i] == "{":
        upper, j = _find_braced(s, i); i = j + 1
    else:
        m = re.match(r"[A-Za-z0-9_]+", s[i:])
        if not m: raise ValueError("Missing/invalid upper bound in sum header")
        upper = m.group(0); i += len(upper)
    if "=" not in lower: raise ValueError("Summation lower bound must be like i=1")
    var, lower_expr = lower.split("=", 1)
    var = var.strip(); lower_expr = lower_expr.strip()
    hdr_end = i
    return hdr_start, hdr_end, var, lower_expr, upper

def _find_first_sum(s: str) -> Optional[Tuple[int, int, str, str, str, str, int]]:
    m = re.search(r"\\sum_", s)
    if not m: return None
    parsed = _parse_sum_header(s, m.start())
    if not parsed: return None
    hdr_start, hdr_end, var, lower_expr, upper_expr = parsed
    # Body
    if hdr_end >= len(s): raise ValueError("Missing summation body after \\sum")
    if s[hdr_end] == "{":
        body, body_end = _find_braced(s, hdr_end); end_after_body = body_end + 1
    else:
        j = hdr_end; depth = 0
        while j < len(s):
            ch = s[j]
            if ch in "([{": depth += 1
            elif ch in ")]}":
                if depth == 0: break
                depth -= 1
            elif ch in "+-" and depth == 0:
                break
            j += 1
        body = s[hdr_end:j].strip(); end_after_body = j
    return hdr_start, hdr_end, var, lower_expr, upper_expr, body, end_after_body

def _rename_keywords_in_expr(expr: str, variables: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """Rename Python keyword variable names in expr to safe aliases (name_)."""
    alias_map: Dict[str, str] = {}
    for k in variables.keys():
        if keyword.iskeyword(k):
            alias_map[k] = k + "_"
    for old, new in alias_map.items():
        expr = re.sub(rf'\b{re.escape(old)}\b', new, expr)
    return expr, alias_map

def _apply_alias_env(env: Dict[str, Any], alias_map: Dict[str, str]) -> Dict[str, Any]:
    """Copy env and add aliases for renamed variables."""
    env2 = dict(env)
    for old, new in alias_map.items():
        if old in env2:
            env2[new] = env2.pop(old)
    return env2

def _eval_one_sum(s: str, variables: Dict[str, float], aggressive: bool = False) -> str:
    occ = _find_first_sum(s)
    if not occ: return s
    hdr_start, hdr_end, var, lower_expr, upper_expr, body, end_after_body = occ

    env = _build_env(variables)

    lower_py = _caret_to_pow(_preprocess_base(lower_expr, aggressive=aggressive))
    upper_py = _caret_to_pow(_preprocess_base(upper_expr, aggressive=aggressive))
    body_py  = _caret_to_pow(_preprocess_base(body,        aggressive=aggressive))

    start_val = int(round(_safe_eval(lower_py, env)))
    end_val   = int(round(_safe_eval(upper_py, env)))

    patt = re.compile(rf"\b([A-Za-z][A-Za-z0-9_]*)_{var}\b")

    total = 0.0
    for k in range(start_val, end_val + 1):
        env_iter = dict(env); env_iter[var] = k
        body_k = patt.sub(rf"\1_{k}", body_py)
        body_k, alias_map = _rename_keywords_in_expr(body_k, variables)
        env_iter = _apply_alias_env(env_iter, alias_map)
        total += float(_safe_eval(body_k, env_iter))

    prefix = s[:hdr_start]; suffix = s[end_after_body:]
    return f"{prefix}({total}){suffix}"

def _expand_all_sums(s: str, variables: Dict[str, float], aggressive: bool = False) -> str:
    prev = None; cur = s; safety = 0
    while prev != cur:
        safety += 1
        if safety > SUM_NEST_LIMIT:
            raise ValueError("Too many nested summations")
        prev = cur
        cur = _eval_one_sum(cur, variables, aggressive=aggressive)
    return cur

def _resolve_free_i_with_N(s: str, variables: Dict[str, float]) -> str:
    if "N" not in variables: return s
    N = int(round(float(variables["N"])))
    return re.sub(r"\b([A-Za-z][A-Za-z0-9_]*)_i\b", rf"\1_{N}", s)

# ----------------------------------------------------------------------
# LaTeX → Python expression
# ----------------------------------------------------------------------
def _preprocess_base(latex: str, aggressive: bool = False) -> str:
    s = _strip_math_delims(latex)

    # Extract main expr and optional trailing constraints ONCE
    main, trailing = _split_top_level_comma_once(s)
    main = _rhs_of_equals(main)  # drop LHS like "X ="

    # Inline constraints (e.g., ", q=1-p") before any other transforms
    main = _apply_trailing_constraints_once(main, trailing)

    # Normal pipeline on the substituted main expression
    s = _replace_text_like(main)
    s = _replace_greek(s)
    s = _resolve_t_subscripts(s, default_t=1)
    s = _normalize_indices(s)
    s = _normalize_basic_ops(s)
    s = _normalize_symbolic_wrappers(s)
    s = _transform_frac(s)
    s = _transform_sqrt(s)
    s = _transform_power_e(s)
    s = _normalize_funcs_to_python(s)
    s = _insert_implicit_mult(s, aggressive=aggressive)
    s = s.strip()
    return s

def latex_to_python_expr(latex: str, variables: Dict[str, float], aggressive: bool = False) -> str:
    s = _preprocess_base(latex, aggressive=aggressive)
    s = _expand_all_sums(s, variables, aggressive=aggressive)
    s = _resolve_free_i_with_N(s, variables)   # handle trailing *_i terms
    s = _caret_to_pow(s)                        # convert ^ and ^{...}
    s = s.replace("{", "(").replace("}", ")")   # neutralize residual braces
    s = re.sub(r"\s+", "", s)
    return s

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
def _build_env(variables: Dict[str, float]) -> Dict[str, Any]:
    env = {"math": math, "max": max, "min": min}
    # Optional alias
    env["ln"] = math.log
    # keep floats for numeric vars; non-numeric will raise cleanly at eval
    for k, v in (variables or {}).items():
        try:
            env[k] = float(v)
        except Exception:
            env[k] = v
    if "e" not in env: env["e"] = math.e
    return env

# ----------------------------------------------------------------------
# Core solver (soft-fail per test, logs only failed tests)
# ----------------------------------------------------------------------
class Sol:
    r"""Evaluate LaTeX formulas with variable maps (no third-party libs)."""

    def __init__(self, tests: List[Dict[str, Any]]):
        self.tests = tests

    def _evaluate_one(self, name: str, ttype: str, formula: str, variables: Dict[str, float]) -> float:
        # ---- PASS 1: safe (no glued-identifier split) ----
        try:
            expr = latex_to_python_expr(formula, variables, aggressive=False)
        except Exception as e2:
            _log_failed_test(name, formula, variables, ttype, phase="transform",
                             kind=type(e2).__name__, msg=str(e2))
            return float(f"{DEFAULT_RESULT:.{ROUND_DP}f}")

        env = _build_env(variables)
        expr1, alias_map1 = _rename_keywords_in_expr(expr, variables)
        env1 = _apply_alias_env(env, alias_map1)
        
        expr1 = _fix_accidental_variable_calls(expr1, env1)
        
        try:
            val = _safe_eval(expr1, env1)
            return float(f"{float(val):.{ROUND_DP}f}")
        except Exception:
            # helpful debug of final expr
            logger.debug(f"[debug] name={name} expr_after_preprocess={expr1}")
            # ---- PASS 2: aggressive (split short glued identifiers like 'bp' -> 'b*p') ----
            try:
                expr2 = latex_to_python_expr(formula, variables, aggressive=True)
                expr2, alias_map2 = _rename_keywords_in_expr(expr2, variables)
                env2 = _apply_alias_env(env, alias_map2)
                
                expr2 = _fix_accidental_variable_calls(expr2, env2)
                
                val2 = _safe_eval(expr2, env2)
                return float(f"{float(val2):.{ROUND_DP}f}")
            except Exception as e2:
                _log_failed_test(name, formula, variables, ttype, phase="eval",
                                 kind=type(e2).__name__, msg=str(e2))
                return float(f"{DEFAULT_RESULT:.{ROUND_DP}f}")

    def solve(self) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for test in self.tests or []:
            name = test.get("name", "")
            ttype = test.get("type", "compute")
            formula = test.get("formula", "")
            variables = test.get("variables", {})
            if not isinstance(variables, dict): variables = {}
            if ttype != "compute":
                _log_failed_test(name, formula, variables, ttype, phase="type",
                                 kind="UnsupportedType", msg=f"type={ttype}")
                out.append({"result": float(f"{DEFAULT_RESULT:.{ROUND_DP}f}")})
                continue
            out.append({"result": self._evaluate_one(name, ttype, formula, variables)})
        return out

# ------------------------- Flask Route -------------------------
@app.route('/trading-formula', methods=['POST'])
def trader():
    """
    POST /trading-formula
    Content-Type: application/json
    Body: JSON array of testcases: [{"name","formula","variables","type": "compute"}, ...]
    Response: 200 + JSON array [{"result": <float rounded to 4 dp>}, ...]
    Soft-fails per testcase and logs only failed ones.
    """
    tests = request.get_json(silent=True)
    if not isinstance(tests, list):
        logger.debug("Payload is not a list — returning empty result set.")
        return jsonify([]), 200

    solver = Sol(tests)
    results = solver.solve()  # guaranteed not to raise
    return jsonify(results), 200
