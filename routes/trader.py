import logging
import math
import re
import ast
from typing import Dict, List, Any

from flask import request, jsonify
from routes import app

logger = logging.getLogger(__name__)

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Num, ast.Constant,
    ast.Name, ast.Load,
    ast.Call, ast.Attribute,
    ast.Tuple, ast.List,
    ast.Subscript, getattr(ast, "Index", ast.slice),  # Py3.12 compat
)

# Explicitly allow these operators
_ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)

def _safe_eval(expr: str, env: dict) -> float:
    """
    Safely evaluate a Python arithmetic expression using a restricted AST.
    Allowed: +,-,*,/,**, unary +/-, max, min, math.* (log, exp, e), numbers & variables.
    """

    def _check(node: ast.AST):
        # BinOp: validate operator, then recurse into left/right
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BIN_OPS):
                raise ValueError(f"Disallowed operator: {type(node.op).__name__}")
            _check(node.left)
            _check(node.right)
            return

        # UnaryOp: validate operator, then recurse
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_OPS):
                raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}")
            _check(node.operand)
            return

        # Function calls: allow names in env (e.g., max/min) or math.<fn>
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in env:
                    raise ValueError(f"Call to unknown function: {func.id}")
            elif isinstance(func, ast.Attribute):
                # allow math.xxx where 'math' is provided in env
                if not (isinstance(func.value, ast.Name) and func.value.id == "math"):
                    raise ValueError("Only math.<func> attribute calls are allowed")
            else:
                raise ValueError("Unsupported callable")
            for a in node.args:
                _check(a)
            for kw in node.keywords or []:
                _check(kw.value)
            return

        # Attribute: only allow math.xxx
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"):
                raise ValueError("Only math.<attr> is allowed")
            return

        # Names, constants, tuples/lists, subscripts are fine — recurse into their children
        if isinstance(node, _ALLOWED_NODES):
            for child in ast.iter_child_nodes(node):
                _check(child)
            return

        raise ValueError(f"Disallowed expression node: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    _check(tree)
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env)
# ------------------------- LaTeX preprocessing helpers -------------------------

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
    # Strip $$...$$ or $...$
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
    # \left \right
    s = s.replace(r"\left", "").replace(r"\right", "")
    # \, thin spaces etc.
    s = s.replace(r"\,", "").replace(r"\;", "").replace(r"\:", "").replace(r"\ ", "")
    return s

def _find_braced(s: str, i: int) -> (str, int):
    """
    Given s and index i at '{', return (content, end_index_of_closing_brace).
    """
    assert s[i] == "{"
    depth = 0
    j = i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                # content between i+1 and j-1
                return s[i+1:j], j
        j += 1
    raise ValueError("Unbalanced braces in LaTeX")

def _transform_frac(s: str) -> str:
    """
    Replace all \frac{A}{B} with ((A)/(B)), recursively handling nested braces.
    """
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
            # Recursively expand inner fracs
            num = _transform_frac(num)
            den = _transform_frac(den)
            out.append(f"(({num})/({den}))")
        else:
            out.append(s[i])
            i += 1
    return "".join(out)

def _transform_power_e(s: str) -> str:
    """
    Handle e^{...} -> (e**(...)) ; then convert remaining '^' to '**'.
    """
    out = []
    i = 0
    while i < len(s):
        if s.startswith("e^{", i):
            # e^{...}
            i += 2  # skip 'e^'
            if i >= len(s) or s[i] != "{":
                raise ValueError("Malformed e^{...}")
            inner, j = _find_braced(s, i)
            i = j + 1
            out.append(f"(e**({inner}))")
        else:
            out.append(s[i])
            i += 1
    s2 = "".join(out)
    # Convert remaining caret powers to Python '**'
    # Be careful not to change things like 'max' etc. Here '^' is safe:
    s2 = s2.replace("^", "**")
    return s2

def _normalize_funcs_to_python(s: str) -> str:
    # Logs: \log(x) or log(x) -> math.log(x)
    s = s.replace(r"\log", "log")
    s = re.sub(r"(?<![A-Za-z0-9_])log\s*\(", "math.log(", s)
    # Exp form: \exp(x) -> math.exp(x)
    s = s.replace(r"\exp", "exp")
    s = re.sub(r"(?<![A-Za-z0-9_])exp\s*\(", "math.exp(", s)
    # Max/Min already normalized to 'max'/'min'
    return s

def _strip_redundant_braces(s: str) -> str:
    # Replace groups like {( ... )} -> ( ... ) in some simple cases (optional)
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


# ------------------------- Summation handling -------------------------

_SUM_PATTERN = re.compile(r"""\\sum_\{([^}]*)\}\^\{([^}]*)\}""")

def _first_sum_occurrence(s: str):
    """
    Find first \sum_{...}^{...} occurrence and return:
    (start_index_of_sum, end_index_of_header, var_name, start_expr, end_expr, body_str, end_index_of_body)
    where body_str is the expression immediately following the header (can be braced or token).
    """
    m = _SUM_PATTERN.search(s)
    if not m:
        return None
    hdr_start = m.start()
    hdr_end = m.end()  # index right after the ^{...}
    sub = m.group(1)   # like i=1
    sup = m.group(2)   # like n

    # parse sub as 'i=1' (allow spaces)
    if "=" not in sub:
        raise ValueError(r"Summation lower bound must be like i=1")
    var, start_expr = sub.split("=", 1)
    var = var.strip()
    start_expr = start_expr.strip()
    end_expr = sup.strip()

    # The body starts at hdr_end. It can be:
    #  - braced {...}
    #  - a single token or parenthesized expr
    # Prefer to capture a braced group if present, else take a token up to a delimiter.
    if hdr_end >= len(s):
        raise ValueError(r"Missing summation body after \sum")
    if s[hdr_end] == "{":
        body, body_end = _find_braced(s, hdr_end)
        end_after_body = body_end + 1
    else:
        # capture until we hit a balanced end: we take a conservative approach—read until next '+' or '-' at same paren depth
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

    # Convert the header expressions (start & end) from LaTeX to Python and evaluate them
    start_py = _preprocess_base(start_expr)
    end_py   = _preprocess_base(end_expr)

    env = _build_env(variables)
    start_val = int(round(_safe_eval(start_py, env)))
    end_val   = int(round(_safe_eval(end_py, env)))

    # Prepare the body: convert LaTeX to Python for each iteration (since body may include \frac etc.)
    body_py_raw = _preprocess_base(body)

    total = 0.0
    for k in range(start_val, end_val + 1):
        env_iter = dict(env)
        env_iter[var] = k
        total += float(_safe_eval(body_py_raw, env_iter))

    # Replace the whole sum occurrence (header + body) with computed numeric literal
    prefix = s[:hdr_start]
    suffix = s[end_after_body:]
    return f"{prefix}({total}){suffix}"

def _expand_all_sums(s: str, variables: Dict[str, float]) -> str:
    # Recursively compute sums until none remain
    prev = None
    cur = s
    safety = 0
    while prev != cur:
        safety += 1
        if safety > 50:
            raise ValueError("Too many nested summations")
        prev = cur
        cur = _eval_one_sum(cur, variables)
    return cur

def _insert_implicit_mult(s: str) -> str:
    """
    Insert missing '*' for implicit multiplication:
      - A( ... ) -> A*( ... )
      - )( or )x -> )*x
      - 2x -> 2*x
      - x y -> x*y
    Protect known calls first so we don't insert '*' before their '('.
    """
    # Protect known call patterns
    protos = {
        "math.log(": "MATHLOG(",
        "math.exp(": "MATHEXP(",
        "max(": "MAXF(",
        "min(": "MINF(",
    }
    for k, v in protos.items():
        s = s.replace(k, v)

    # 1) var/number/')' followed by '('  -> insert '*'
    s = re.sub(r'([0-9A-Za-z_)\]])\s*\(', r'\1*(', s)

    # 2) ')' followed by var/number  -> insert '*'
    s = re.sub(r'\)\s*([0-9A-Za-z_])', r')*\1', s)

    # 3) number followed by variable  -> insert '*'
    s = re.sub(r'(\d)\s*([A-Za-z_])', r'\1*\2', s)

    # 4) variable followed by variable  -> insert '*'
    s = re.sub(r'([A-Za-z_][A-Za-z_0-9]*)\s+([A-Za-z_])', r'\1*\2', s)

    # Unprotect
    for k, v in protos.items():
        s = s.replace(v, k)

    return s


# ------------------------- Main latex → python expression -------------------------

def latex_to_python_expr(latex: str, variables: Dict[str, float]) -> str:
    """
    Turn a LaTeX RHS into a Python arithmetic string ready for safe eval.
    Handles: +,-,*,/, **, \cdot, \times, \frac, \max, \min, \sum, e^x, log/exp.
    """
    s = _preprocess_base(latex)
    # Expand all \sum occurrences by computing them with current variables
    s = _expand_all_sums(s, variables)
    # Final cleanup: multiple spaces -> none
    s = re.sub(r"\s+", "", s)
    return s


# ------------------------- Core Solver -------------------------

class Sol:
    """
    Evaluate an array of LaTeX formulas with variable maps (without third-party libs).

    Input test case:
      {
        "name": "...",
        "formula": "...",   # LaTeX (may include $$..$$ or leading 'X = ...')
        "variables": {...}, # dict of str -> float
        "type": "compute"
      }

    Output: list like [{"result": 12.3456}, ...] with rounding to 4 decimals.
    """

    def __init__(self, tests: List[Dict[str, Any]]):
        self.tests = tests

    def _evaluate_one(self, formula: str, variables: Dict[str, float]) -> float:
        expr = latex_to_python_expr(formula, variables)
        env = _build_env(variables)
        val = _safe_eval(expr, env)
        return float(f"{float(val):.4f}")

    def solve(self) -> List[Dict[str, float]]:
        out = []
        for test in self.tests:
            if test.get("type", "compute") != "compute":
                raise ValueError(f"Unsupported type '{test.get('type')}' in test '{test.get('name')}'")
            vars_obj = test.get("variables", {})
            if not isinstance(vars_obj, dict):
                raise ValueError(f"'variables' must be an object in test '{test.get('name')}'")
            res = self._evaluate_one(test.get("formula", ""), vars_obj)
            out.append({"result": res})
        return out


def _build_env(variables: Dict[str, float]) -> Dict[str, Any]:
    """
    Build the safe evaluation environment:
    - variables as provided
    - math module (for log, exp, etc.)
    - max/min
    - e constant (unless user overrides with a variable named 'e')
    """
    env = {"math": math, "max": max, "min": min}
    env.update({k: float(v) for k, v in variables.items()})
    if "e" not in env:
        env["e"] = math.e
    return env


# ------------------------- Flask Route -------------------------

@app.route('/trading-formula', methods=['POST'])
def trader():
    """
    Expected request: application/json (array of test cases)
    Returns: application/json (array of {"result": number})
    """
    tests = request.get_json(silent=True)
    if not isinstance(tests, list):
        return jsonify(error="Payload must be a JSON array of test cases"), 400

    try:
        solver = Sol(tests)
        results = solver.solve()
        return jsonify(results), 200
    except Exception as e:
        # logger.exception("Failed to evaluate formulas")
        return jsonify(error=str(e)), 400
