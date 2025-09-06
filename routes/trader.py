import logging
import math
import re
import ast
import json
from typing import Dict, List, Any, Tuple, Optional
import keyword 

from flask import request, jsonify
from routes import app
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------------
# Logging: only failed testcases
# ----------------------------------------------------------------------
# NOTE: By default we log to stdout (so PaaS logs pick it up).
# If you prefer a file, add a FileHandler to FAIL_LOGGER.
FAIL_LOGGER = logging.getLogger(__name__)
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
        "error": {"phase": phase, "kind": kind, "msg": msg[:200]}
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
            for kw in node.keywords or []: _check(kw.value)
            return
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"):
                raise ValueError("Only math.<attr> is allowed")
            return
        if isinstance(node, _ALLOWED_NODES):
            for child in ast.iter_child_nodes(node): _check(child)
            return
        raise ValueError(f"Disallowed expression node: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    _check(tree)
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env)

# ----------------------------------------------------------------------
# LaTeX preprocessing
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
def _drop_trailing_constraints(s: str) -> str:
    """Drop anything after a top-level comma: 'expr , something' -> 'expr'."""
    depth = 0
    for idx, ch in enumerate(s):
        if ch in "([{": depth += 1
        elif ch in ")]}": depth -= 1 if depth > 0 else 0
        elif ch == "," and depth == 0:
            return s[:idx].strip()
    return s
def _rename_keywords_in_expr(expr: str, variables: Dict[str, Any]) -> tuple[str, Dict[str, str]]:
    """Rename Python keyword variable names in expr to safe aliases (name_)."""
    alias_map: Dict[str, str] = {}
    for k in variables.keys():
        if keyword.iskeyword(k):
            alias_map[k] = k + "_"
    # Also cover Greek-mapped names that might collide (e.g., lambda)
    for k in list(alias_map.keys()):
        pass  # (already covered via keyword.iskeyword)

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

def _resolve_t_subscripts(s: str, default_t: int = 1) -> str:
    """Turn _{t-k} -> _<default_t-k>, and _{t} -> _<default_t>.  (Before we strip braces.)"""
    def repl_minus(m):
        k = int(m.group(1))
        return f"_{max(default_t - k, 0)}"
    s = re.sub(r"_\{t-([0-9]+)\}", repl_minus, s)
    s = re.sub(r"_\{t\}", f"_{default_t}", s)
    return s

def _find_paren_group(s: str, i: int) -> tuple[str, int]:
    """Given s[i]=='(', return (content, end_index)."""
    assert s[i] == "("
    depth = 0
    j = i
    while j < len(s):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return s[i+1:j], j
        j += 1
    raise ValueError("Unbalanced parentheses")

def _split_top_level_commas(s: str) -> list[str]:
    parts, depth, buf = [], 0, []
    for ch in s:
        if ch in "([{": depth += 1
        elif ch in ")]}": depth -= 1 if depth > 0 else 0
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return parts

def _clean_ident_piece(x: str) -> str:
    # Keep letters, digits, underscore; drop everything else (space, *, +, - …)
    return re.sub(r"[^A-Za-z0-9_]+", "", x)

def _normalize_symbolic_wrappers(s: str) -> str:
    """
    Convert Cov(X,Y) -> Cov_X_Y and Var(X) -> Var_X so they map to JSON keys
    like Cov_DeltaS_DeltaF, Var_r_i, Var_DeltaF.
    """
    out = []
    i = 0
    while i < len(s):
        if s.startswith("Cov(", i) or s.startswith("Var(", i):
            name = "Cov" if s.startswith("Cov(", i) else "Var"
            i0 = i + len(name)
            content, j = _find_paren_group(s, i0)
            i = j + 1
            if name == "Cov":
                args = _split_top_level_commas(content)
                if len(args) != 2:
                    # fallback: just squash everything
                    inner = _clean_ident_piece(content)
                    out.append(f"{name}_{inner}")
                else:
                    a = _clean_ident_piece(args[0])
                    b = _clean_ident_piece(args[1])
                    out.append(f"{name}_{a}_{b}")
            else:  # Var
                inner = _clean_ident_piece(content)
                out.append(f"{name}_{inner}")
        else:
            out.append(s[i]); i += 1
    return "".join(out)

def _caret_to_pow(s: str) -> str:
    """
    Convert a^{b} -> (a**(b)) and then any remaining '^' -> '**'.
    """
    out = []
    i = 0
    while i < len(s):
        if s[i] == "^":
            # look back for a base token (greedy)
            base_start = i - 1
            while base_start >= 0 and re.match(r"[A-Za-z0-9_\)\]]", s[base_start]):
                base_start -= 1
            base = s[base_start+1:i]
            i += 1
            if i < len(s) and s[i] == "{":
                inner, j = _find_braced(s, i)
                i = j + 1
                out.append(f"({base}**({inner}))")
            else:
                # simple exponent token (name/number/parenthesized)
                # collect a simple name/number
                m = re.match(r"[A-Za-z0-9_]+", s[i:])
                if m:
                    exp_tok = m.group(0)
                    i += len(exp_tok)
                    out.append(f"({base}**{exp_tok})")
                else:
                    out.append(base + "**")  # let later parse complain if malformed
        else:
            out.append(s[i]); i += 1
    s2 = "".join(out)
    return s2.replace("^", "**")  # final safety

def _strip_math_delims(s: str) -> str:
    s = s.strip()
    if s.startswith("$$") and s.endswith("$$"): return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$"):  return s[1:-1].strip()
    return s

def _rhs_of_equals(s: str) -> str:
    return s.split("=", 1)[1].strip() if "=" in s else s

def _replace_text_vars(s: str) -> str:
    # \text{Trade Amount} -> TradeAmount
    def joiner(m): return re.sub(r"\s+", "", m.group(1))
    return re.sub(r"\\text\{([^}]*)\}", joiner, s)

def _replace_greek(s: str) -> str:
    for k, v in _GREEK.items(): s = s.replace(k, v)
    return s

def _normalize_indices(s: str) -> str:
    # E[R_m] -> E_R_m ; A[B] -> A_B
    s = re.sub(r"([A-Za-z]+)\[([^\]]+)\]", lambda m: f"{m.group(1)}_{m.group(2)}", s)
    # X_{a_b} -> X_a_b ; remove braces around subscripts
    s = re.sub(r"_\{([^}]+)\}", lambda m: "_" + m.group(1), s)
    # Collapse spaces around underscores
    s = re.sub(r"\s*_\s*", "_", s)
    return s

def _normalize_basic_ops(s: str) -> str:
    # Multiplication
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    # Max/Min
    s = s.replace(r"\operatorname{Max}", "max").replace(r"\operatorname{Min}", "min")
    s = s.replace(r"\max", "max").replace(r"\min", "min")
    # Remove size/spacing macros and \left/\right
    for tok in [r"\left", r"\right", r"\Big", r"\big", r"\Bigg", r"\bigg", r"\!", r"\,", r"\;", r"\:", r"\ "]:
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
    """Handle e^{...} -> (e**(...)). We convert other '^' later, after sums."""
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
    # \log(x) -> math.log(x) | \exp(x) -> math.exp(x)
    s = s.replace(r"\log", "log").replace(r"\exp", "exp")
    s = re.sub(r"(?<![A-Za-z0-9_])log\s*\(", "math.log(", s)
    s = re.sub(r"(?<![A-Za-z0-9_])exp\s*\(", "math.exp(", s)
    return s

def _insert_implicit_mult(s: str, aggressive: bool = False) -> str:
    """
    Insert '*' for implicit multiplication, while protecting real calls.
    'aggressive=False' avoids splitting complex identifiers (e.g., E_R_m).
    'aggressive=True' additionally splits glued identifiers like bp -> b*p.
    """
    protos = {
        "math.log(": "MATHLOG⟬",
        "math.exp(": "MATHEXP⟬",
        "max(": "MAXF⟬",
        "min(": "MINF⟬",
    }
    for k, v in protos.items():
        s = s.replace(k, v)

    # 1) A( -> A*(
    s = re.sub(r'([0-9A-Za-z_)\]])\s*\(', r'\1*(', s)
    # 2) )( or )x -> )*x
    s = re.sub(r'\)\s*([0-9A-Za-z_])', r')*\1', s)
    # 3) 2x -> 2*x
    s = re.sub(r'(\d)\s*([A-Za-z_])', r'\1*\2', s)
    # 4) x y -> x*y
    s = re.sub(r'([A-Za-z_][A-Za-z_0-9]*)\s+([A-Za-z_])', r'\1*\2', s)

    if aggressive:
        # 5) xy (adjacent identifiers) -> x*y, but DON'T split if the left ends with '_'
        #    e.g., 'bp' -> 'b* p'   but 'E_R_m' stays intact
        s = re.sub(r'([A-Za-z_][A-Za-z0-9]*[A-Za-z0-9])(?=[A-Za-z])', r'\1*', s)

    for k, v in protos.items():
        s = s.replace(v, k)
    return s

def _strip_redundant_braces(s: str) -> str:
    return s

def _preprocess_base(latex: str, aggressive: bool = False) -> str:
    s = _insert_implicit_mult(s, aggressive=aggressive)
    s = _strip_math_delims(latex)
    s = _rhs_of_equals(s)
    s = _drop_trailing_constraints(s)        # NEW: drop ", q=1-p" style tails
    s = _replace_text_vars(s)
    s = _replace_greek(s)
    s = _resolve_t_subscripts(s, default_t=1)  # NEW: handle _{t-1}, _{t}
    s = _normalize_indices(s)
    s = _normalize_basic_ops(s)
    s = _normalize_symbolic_wrappers(s)      # NEW: Cov(...), Var(...) -> variable ids
    s = _transform_frac(s)
    s = _transform_sqrt(s)
    s = _transform_power_e(s)                # e^{...}
    s = _normalize_funcs_to_python(s)        # log/exp -> math.log/exp
    s = _insert_implicit_mult(s)
    s = _strip_redundant_braces(s)
    return s.strip()


def _caret_to_pow(s: str) -> str:
    """Convert remaining '^' to Python '**' AFTER sums are expanded."""
    return s.replace("^", "**")

# ----------------------------------------------------------------------
# Summation (\sum) support (accepts \sum_{i=1}^{N} and \sum_i=1^N)
# ----------------------------------------------------------------------
def _drop_trailing_constraints(s: str) -> str:
    """Drop anything after a top-level comma: 'expr , something' -> 'expr'."""
    depth = 0
    for idx, ch in enumerate(s):
        if ch in "([{": depth += 1
        elif ch in ")]}": depth -= 1 if depth > 0 else 0
        elif ch == "," and depth == 0:
            return s[:idx].strip()
    return s

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

    # identifiers like CF_t, PD_i, Var_r_i → replace final _<var> with _<k>
    patt = re.compile(rf"\b([A-Za-z][A-Za-z0-9_]*)_{var}\b")

    total = 0.0
    for k in range(start_val, end_val + 1):
        env_iter = dict(env); env_iter[var] = k
        body_k = patt.sub(rf"\1_{k}", body_py)
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
def latex_to_python_expr(latex: str, variables: Dict[str, float], aggressive: bool = False) -> str:
    s = _preprocess_base(latex, aggressive=aggressive)
    s = _expand_all_sums(s, variables, aggressive=aggressive)
    s = _resolve_free_i_with_N(s, variables)
    s = _caret_to_pow(s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\s+", "", s)
    return s


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
def _build_env(variables: Dict[str, float]) -> Dict[str, Any]:
    env = {"math": math, "max": max, "min": min}
    env.update({k: float(v) for k, v in variables.items()})
    if "e" not in env: env["e"] = math.e
    return env

# ----------------------------------------------------------------------
# Core solver (soft-fail per test, only logs failed tests)
# ----------------------------------------------------------------------
class Sol:
    r"""Evaluate LaTeX formulas with variable maps (no third-party libs)."""

    def __init__(self, tests: List[Dict[str, Any]]):
        self.tests = tests

    def _evaluate_one(self, name: str, ttype: str, formula: str, variables: Dict[str, float]) -> float:
        # ---- PASS 1: safe (no glued-identifier split) ----
        try:
            expr = latex_to_python_expr(formula, variables, aggressive=False)
        except Exception as e:
            # If transform fails, try aggressive transform before giving up
            try:
                expr = latex_to_python_expr(formula, variables, aggressive=True)
            except Exception as e2:
                _log_failed_test(name, formula, variables, ttype, phase="transform",
                                kind=type(e2).__name__, msg=str(e2))
                return float(f"{DEFAULT_RESULT:.{ROUND_DP}f}")

        env = _build_env(variables)
        expr1, alias_map1 = _rename_keywords_in_expr(expr, variables)
        env1 = _apply_alias_env(env, alias_map1)
        try:
            val = _safe_eval(expr1, env1)
            return float(f"{float(val):.{ROUND_DP}f}")
        except Exception:
            # ---- PASS 2: aggressive (split glued identifiers like 'bp' -> 'b*p') ----
            try:
                expr2 = latex_to_python_expr(formula, variables, aggressive=True)
                expr2, alias_map2 = _rename_keywords_in_expr(expr2, variables)
                env2 = _apply_alias_env(env, alias_map2)
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
    Accepts: JSON array of testcases.
    Returns: 200 + JSON array of {"result": number} for each testcase.
    This endpoint NEVER aborts the whole batch due to a single bad testcase.
    """
    tests = request.get_json(silent=True)
    if not isinstance(tests, list):
        # Soft behavior: if payload is not a list, respond with empty result list (still 200)
        logger.debug("Payload is not a list — returning empty result set.")
        return jsonify([]), 200

    solver = Sol(tests)
    results = solver.solve()  # guaranteed not to raise
    return jsonify(results), 200
