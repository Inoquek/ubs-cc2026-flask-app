import logging
from functools import lru_cache
from flask import request, jsonify
from routes import app

ATTACK_MINUTES = 10
COOLDOWN_MINUTES = 10

logger = logging.getLogger(__name__)

def solve_case(case):
    intel = case["intel"]
    reserve = case["reserve"]
    fronts = case["fronts"]      # kept for parity; not used by DP logic
    stamina_max = case["stamina"]

    n = len(intel)
    F = [f for f, _ in intel]
    C = [c for _, c in intel]

    @lru_cache(maxsize=None)
    def dp(i: int, mp: int, stamina: int, chain: int) -> int:
        """
        Minimal remaining time (minutes) from index i with (mp, stamina).
        chain == 1 : previous action was an attack and same front as F[i] -> extend next attack (0 min)
        chain == 0 : no extend available (e.g., after cooldown or different front)
        """
        if i == n:
            # Mandatory final cooldown so Klein is ready to reinforce
            return COOLDOWN_MINUTES

        front_i = F[i]
        cost_i = C[i]

        best = float("inf")

        # --- Option 1: cooldown (only if it changes state to avoid self-recursion)
        if not (mp == reserve and stamina == stamina_max and chain == 0):
            best = min(best, COOLDOWN_MINUTES + dp(i, reserve, stamina_max, 0))

        # --- Option 2: attack now (if resources allow)
        if cost_i <= mp and stamina > 0:
            # Attack cost: 0 if extend allowed, else 10
            attack_cost = 0 if chain == 1 else ATTACK_MINUTES

            # Next state's chain is 1 only if the *next* front equals current front
            if i + 1 < n and F[i + 1] == front_i:
                next_chain = 1
            else:
                next_chain = 0

            best = min(
                best,
                attack_cost + dp(i + 1, mp - cost_i, stamina - 1, next_chain),
            )

        return best

    total_time = dp(0, reserve, stamina_max, 0)
    return {"time": total_time}

@app.route("/the-mages-gambit", methods=["POST"])
def gambit():
    if not request.is_json:
        return jsonify({"error": "Expected application/json body"}), 400
    payload = request.get_json()

    if not isinstance(payload, list):
        return jsonify({"error": "Expected a JSON array of cases"}), 400

    try:
        result = [solve_case(case) for case in payload]
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result), 200