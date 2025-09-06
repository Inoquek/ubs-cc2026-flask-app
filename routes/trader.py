import logging
import math
import re
import ast
import json
from typing import Dict, List, Any, Tuple, Optional

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

def _insert_implicit_mult(s: str) -> str:
    """
    Insert '*' for implicit multiplication, while protecting true calls.
    Uses a sentinel '⟬' to temporarily hide '(' from known calls so the
    regex won't insert '*' between name and '('.
    """
    protos = {
        "math.log(": "MATHLOG⟬",
        "math.exp(": "MATHEXP⟬",
        "max(": "MAXF⟬",
        "min(": "MINF⟬",
    }
    for k, v in protos.items(): s = s.replace(k, v)

    # A( -> A*( |  )( or )x -> )*x | 2x -> 2*x | x y -> x*y | (a+b)(c+d) -> (a+b)*(c+d)
    s = re.sub(r'([0-9A-Za-z_)\]])\s*\(', r'\1*(', s)
    s = re.sub(r'\)\s*([0-9A-Za-z_])', r')*\1', s)
    s = re.sub(r'(\d)\s*([A-Za-z_])', r'\1*\2', s)
    s = re.sub(r'([A-Za-z_][A-Za-z_0-9]*)\s+([A-Za-z_])', r'\1*\2', s)

    for k, v in protos.items(): s = s.replace(v, k)
    return s

def _strip_redundant_braces(s: str) -> str:
    return s

def _preprocess_base(latex: str) -> str:
    s = _strip_math_delims(latex)
    s = _rhs_of_equals(s)
    s = _replace_text_vars(s)
    s = _replace_greek(s)
    s = _normalize_indices(s)
    s = _normalize_basic_ops(s)
    s = _transform_frac(s)
    s = _transform_sqrt(s)
    s = _transform_power_e(s)     # only e^{...}
    s = _normalize_funcs_to_python(s)
    s = _insert_implicit_mult(s)
    s = _strip_redundant_braces(s)
    return s.strip()

def _caret_to_pow(s: str) -> str:
    """Convert remaining '^' to Python '**' AFTER sums are expanded."""
    return s.replace("^", "**")

# ----------------------------------------------------------------------
# Summation (\sum) support (accepts \sum_{i=1}^{N} and \sum_i=1^N)
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

def _eval_one_sum(s: str, variables: Dict[str, float]) -> str:
    occ = _find_first_sum(s)
    if not occ: return s
    hdr_start, hdr_end, var, lower_expr, upper_expr, body, end_after_body = occ
    env = _build_env(variables)
    lower_py = _caret_to_pow(_preprocess_base(lower_expr))
    upper_py = _caret_to_pow(_preprocess_base(upper_expr))
    body_py  = _caret_to_pow(_preprocess_base(body))
    start_val = int(round(_safe_eval(lower_py, env)))
    end_val   = int(round(_safe_eval(upper_py, env)))
    total = 0.0
    for k in range(start_val, end_val + 1):
        env_iter = dict(env); env_iter[var] = k
        total += float(_safe_eval(body_py, env_iter))
    prefix = s[:hdr_start]; suffix = s[end_after_body:]
    return f"{prefix}({total}){suffix}"

def _expand_all_sums(s: str, variables: Dict[str, float]) -> str:
    prev = None; cur = s; safety = 0
    while prev != cur:
        safety += 1
        if safety > SUM_NEST_LIMIT:
            raise ValueError("Too many nested summations")
        prev = cur
        cur = _eval_one_sum(cur, variables)
    return cur

# ----------------------------------------------------------------------
# LaTeX → Python expression
# ----------------------------------------------------------------------
def latex_to_python_expr(latex: str, variables: Dict[str, float]) -> str:
    s = _preprocess_base(latex)          # may raise
    s = _expand_all_sums(s, variables)   # may raise
    s = _caret_to_pow(s)
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
        try:
            expr = latex_to_python_expr(formula, variables)
        except Exception as e:
            _log_failed_test(name, formula, variables, ttype, phase="transform",
                             kind=type(e).__name__, msg=str(e))
            return float(f"{DEFAULT_RESULT:.{ROUND_DP}f}")

        env = _build_env(variables)
        try:
            val = _safe_eval(expr, env)
            return float(f"{float(val):.{ROUND_DP}f}")
        except Exception as e:
            _log_failed_test(name, formula, variables, ttype, phase="eval",
                             kind=type(e).__name__, msg=str(e))
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

# ----------------------------------------------------------------------
# Flask route
# ----------------------------------------------------------------------
@app.route('/trading-formula', methods=['POST'])
def trading_formula():
    """
    Accepts: JSON array of testcases.
    Returns: 200 + JSON array of {"result": number} for each testcase (never aborts the batch).
    Only failed testcases are logged, as a single JSON line each.
    """
    tests = request.get_json(silent=True)
    if not isinstance(tests, list):
        # soft return: empty set, no logs
        return jsonify([]), 200

    solver = Sol(tests)
    results = solver.solve()
    return jsonify(results), 200




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
