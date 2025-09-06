import logging
import math
import re
import ast
from typing import Dict, List, Any

from flask import request, jsonify
from routes import app

logger = logging.getLogger(__name__)

# ------------------------- Config -------------------------
DEFAULT_RESULT = 0.0  # what we return for a single test that fails
ROUND_DP = 4          # round to 4 decimals


# ------------------------- Safe evaluation -------------------------
# Allowed node types
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num, ast.Constant,
    ast.Name, ast.Load,
    ast.Call, ast.Attribute,
    ast.Tuple, ast.List,
    ast.Subscript, getattr(ast, "Index", ast.slice),
)

# Allowed ops
_ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)

def _safe_eval(expr: str, env: Dict[str, Any]) -> float:
    """
    Safely evaluate a Python arithmetic expression using a restricted AST.
    Allowed: +,-,*,/,**, unary +/-, max, min, math.* (log, exp, e), numbers & variables.
    """

    def _check(node: ast.AST):
        # Validate operators on parents; don't descend into op nodes themselves.
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BIN_OPS):
                raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
            _check(node.left)
            _check(node.right)
            return

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_OPS):
                raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")
            _check(node.operand)
            return

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in env:
                    raise ValueError(f"Call to unknown function: {func.id}")
            elif isinstance(func, ast.Attribute):
                # Only allow math.xxx
                if not (isinstance(func.value, ast.Name) and func.value.id == "math"):
                    raise ValueError("Only math.<func> attribute calls are allowed")
            else:
                raise ValueError("Unsupported callable")
            for a in node.args:
                _check(a)
            for kw in node.keywords or []:
                _check(kw.value)
            return

        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"):
                raise ValueError("Only math.<attr> is allowed")
            return

        if isinstance(node, _ALLOWED_NODES):
            for child in ast.iter_child_nodes(node):
                _check(child)
            return

        raise ValueError(f"Disallowed expression node: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    _check(tree)
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env)


# ------------------------- LaTeX preprocessing -------------------------
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

_SUM_PATTERN = re.compile(r"""\\sum_\{([^}]*)\}\^\{([^}]*)\}""")

def _strip_math_delims(s: str) -> str:
    s = s.strip()
    if s.startswith("$$") and s.endswith("$$"):
        return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$"):
        return s[1:-1].strip()
    return s

def _rhs_of_equals(s: str) -> str:
    return s.split("=", 1)[1].strip() if "=" in s else s

def _replace_text_vars(s: str) -> str:
    # \text{Trade Amount} -> TradeAmount
    def joiner(m):
        return re.sub(r"\s+", "", m.group(1))
    return re.sub(r"\\text\{([^}]*)\}", joiner, s)

def _replace_greek(s: str) -> str:
    for k, v in _GREEK.items():
        s = s.replace(k, v)
    return s

def _normalize_indices(s: str) -> str:
    # E[R_m] -> E_R_m ; A[B] -> A_B
    s = re.sub(r"([A-Za-z]+)\[([^\]]+)\]", lambda m: f"{m.group(1)}_{m.group(2)}", s)
    # X_{a_b} -> X_a_b ; remove braces around subscripts
    s = re.sub(r"_\{([^}]+)\}", lambda m: "_" + m.group(1), s)
    # Remove spaces around underscores: Z_ alpha -> Z_alpha
    s = re.sub(r"\s*_\s*", "_", s)
    return s

def _normalize_basic_ops(s: str) -> str:
    # Multiplication
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    # Max/min
    s = s.replace(r"\operatorname{Max}", "max").replace(r"\operatorname{Min}", "min")
    s = s.replace(r"\max", "max").replace(r"\min", "min")
    # Remove \left \right and spacing macros
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\,", "").replace(r"\;", "").replace(r"\:", "").replace(r"\ ", "")
    return s

def _find_braced(s: str, i: int) -> (str, int):
    assert s[i] == "{"
    depth = 0
    j = i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i+1:j], j
        j += 1
    raise ValueError("Unbalanced braces in LaTeX")

def _transform_frac(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if s.startswith(r"\frac", i):
            i += len(r"\frac")
            if i >= len(s) or s[i] != "{":
                raise ValueError(r"Expected '{' after \frac")
            num, j = _find_braced(s, i)
            i = j + 1
            if i >= len(s) or s[i] != "{":
                raise ValueError(r"Expected '{' for denominator in \frac")
            den, j = _find_braced(s, i)
            i = j + 1
            num = _transform_frac(num)
            den = _transform_frac(den)
            out.append(f"(({num})/({den}))")
        else:
            out.append(s[i])
            i += 1
    return "".join(out)

def _transform_power_e(s: str) -> str:
    # e^{...} -> (e**(...)); then '^' -> '**'
    out = []
    i = 0
    while i < len(s):
        if s.startswith("e^{", i):
            i += 2
            if i >= len(s) or s[i] != "{":
                raise ValueError("Malformed e^{...}")
            inner, j = _find_braced(s, i)
            i = j + 1
            out.append(f"(e**({inner}))")
        else:
            out.append(s[i])
            i += 1
    s2 = "".join(out).replace("^", "**")
    return s2

def _normalize_funcs_to_python(s: str) -> str:
    # \log(x) -> math.log(x), \exp(x) -> math.exp(x)
    s = s.replace(r"\log", "log").replace(r"\exp", "exp")
    s = re.sub(r"(?<![A-Za-z0-9_])log\s*\(", "math.log(", s)
    s = re.sub(r"(?<![A-Za-z0-9_])exp\s*\(", "math.exp(", s)
    return s

def _insert_implicit_mult(s: str) -> str:
    """
    Insert '*' for implicit multiplication, while protecting real calls.
    """
    # Protect known function-call prefixes so we don't insert before '('
    protos = {
        "math.log(": "MATHLOG(",
        "math.exp(": "MATHEXP(",
        "max(": "MAXF(",
        "min(": "MINF(",
    }
    for k, v in protos.items():
        s = s.replace(k, v)

    # A( -> A*( ;  )( or )x -> )*x ; 2x -> 2*x ; x y -> x*y
    s = re.sub(r'([0-9A-Za-z_)\]])\s*\(', r'\1*(', s)
    s = re.sub(r'\)\s*([0-9A-Za-z_])', r')*\1', s)
    s = re.sub(r'(\d)\s*([A-Za-z_])', r'\1*\2', s)
    s = re.sub(r'([A-Za-z_][A-Za-z_0-9]*)\s+([A-Za-z_])', r'\1*\2', s)

    for k, v in protos.items():
        s = s.replace(v, k)
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
    s = _transform_power_e(s)
    s = _normalize_funcs_to_python(s)
    s = _insert_implicit_mult(s)
    s = _strip_redundant_braces(s)
    return s.strip()


# ------------------------- Summation support -------------------------
def _first_sum_occurrence(s: str):
    m = _SUM_PATTERN.search(s)
    if not m:
        return None
    hdr_start = m.start()
    hdr_end = m.end()
    sub = m.group(1)   # like i=1
    sup = m.group(2)   # like n

    if "=" not in sub:
        raise ValueError(r"Summation lower bound must be like i=1")
    var, start_expr = sub.split("=", 1)
    var = var.strip()
    start_expr = start_expr.strip()
    end_expr = sup.strip()

    if hdr_end >= len(s):
        raise ValueError(r"Missing summation body after \sum")
    if s[hdr_end] == "{":
        body, body_end = _find_braced(s, hdr_end)
        end_after_body = body_end + 1
    else:
        j = hdr_end
        depth = 0
        while j < len(s):
            ch = s[j]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch in "+-" and depth == 0:
                break
            j += 1
        body = s[hdr_end:j].strip()
        end_after_body = j

    return (hdr_start, hdr_end, var, start_expr, end_expr, body, end_after_body)

def _eval_one_sum(s: str, variables: Dict[str, float]) -> str:
    occ = _first_sum_occurrence(s)
    if not occ:
        return s
    hdr_start, hdr_end, var, start_expr, end_expr, body, end_after_body = occ

    # Safely evaluate bounds with current variables
    env = _build_env(variables)
    start_py = _preprocess_base(start_expr)
    end_py   = _preprocess_base(end_expr)
    start_val = int(round(_safe_eval(start_py, env)))
    end_val   = int(round(_safe_eval(end_py, env)))

    body_py_raw = _preprocess_base(body)
    total = 0.0
    for k in range(start_val, end_val + 1):
        env_iter = dict(env)
        env_iter[var] = k
        total += float(_safe_eval(body_py_raw, env_iter))

    prefix = s[:hdr_start]
    suffix = s[end_after_body:]
    return f"{prefix}({total}){suffix}"

def _expand_all_sums(s: str, variables: Dict[str, float]) -> str:
    prev = None
    cur = s
    safety = 0
    while prev != cur:
        safety += 1
        if safety > 50:
            break  # soft-fail: stop expanding rather than raising
        prev = cur
        try:
            cur = _eval_one_sum(cur, variables)
        except Exception as e:
            logger.debug("sum expansion failed: %s", e)
            break  # soft-fail
    return cur


# ------------------------- LaTeX → Python expr -------------------------
def latex_to_python_expr(latex: str, variables: Dict[str, float]) -> str:
    s = _preprocess_base(latex)
    s = _expand_all_sums(s, variables)
    s = re.sub(r"\s+", "", s)
    return s


# ------------------------- Environment -------------------------
def _build_env(variables: Dict[str, float]) -> Dict[str, Any]:
    env = {"math": math, "max": max, "min": min}
    env.update({k: float(v) for k, v in variables.items()})
    if "e" not in env:
        env["e"] = math.e
    return env


# ------------------------- Core Solver (SOFT FAIL PER TEST) -------------------------
class Sol:
    r"""
    Evaluate an array of LaTeX formulas with variable maps (no third-party libs).
    Per-test soft failure: on any error, returns DEFAULT_RESULT rounded to 4dp.
    """

    def __init__(self, tests: List[Dict[str, Any]]):
        self.tests = tests

    def _evaluate_one(self, name: str, formula: str, variables: Dict[str, float]) -> float:
        try:
            expr = latex_to_python_expr(formula, variables)
            env = _build_env(variables)
            val = _safe_eval(expr, env)
            return float(f"{float(val):.{ROUND_DP}f}")
        except Exception as e:
            logger.debug("Test '%s' failed (%s). Returning default.", name, e, exc_info=True)
            return float(f"{DEFAULT_RESULT:.{ROUND_DP}f}")

    def solve(self) -> List[Dict[str, float]]:  # NEVER raises
        out: List[Dict[str, float]] = []
        for test in self.tests or []:
            name = test.get("name", "")
            ttype = test.get("type", "compute")
            formula = test.get("formula", "")
            variables = test.get("variables", {})
            if not isinstance(variables, dict):
                variables = {}

            if ttype != "compute":
                logger.debug("Unsupported type '%s' in test '%s' — returning default.", ttype, name)
                out.append({"result": float(f"{DEFAULT_RESULT:.{ROUND_DP}f}")})
                continue

            out.append({"result": self._evaluate_one(name, formula, variables)})
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
