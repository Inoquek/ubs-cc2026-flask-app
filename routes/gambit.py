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
    fronts = case["fronts"]  # kept for parity; not needed by DP logic itself
    stamina_max = case["stamina"]

    n = len(intel)

    # Pre-extract fronts & costs for speed/clarity
    F = [f for f, _ in intel]
    C = [c for _, c in intel]

    @lru_cache(maxsize=None)
    def dp(i: int, mp: int, stamina: int, chain: int) -> int:
        """
        Returns the minimal remaining time (in minutes) from position i with the given resources.
        chain = 1 means: the previous action was an attack on the same front as intel[i].front,
                         so an immediate attack now would be an 'extend' costing 0.
               = 0 means: no extend available (either previous was cooldown or different front).
        """
        if i == n:
            # Mandatory final cooldown so Klein is ready to reinforce
            return COOLDOWN_MINUTES

        front_i = F[i]
        cost_i = C[i]

        best = float('inf')

        # Option 1: Cooldown first (always allowed)
        if not (mp == reserve and stamina == stamina_max and chain == 0):
            best = min(best, COOLDOWN_MINUTES + dp(i, reserve, stamina_max, 0))

        # Option 2: Attack now (only if resources allow)
        if cost_i <= mp and stamina > 0:
            # time to attack: 0 if extend chain, else 10
            attack_cost = 0 if chain == 1 else ATTACK_MINUTES

            # After attacking i, we move to i+1. The *next* state's chain is 1
            # only if the next front equals current front.
            if i + 1 < n and F[i + 1] == front_i:
                next_chain = 1
            else:
                next_chain = 0

            best = min(
                best,
                attack_cost + dp(i + 1, mp - cost_i, stamina - 1, next_chain)
            )

        return best

    # Start at i=0 with full resources and no active chain.
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
if not (mp == reserve and stamina == stamina_max and chain == 0):
    best = min(best, COOLDOWN_MINUTES + dp(i, reserve, stamina_max, 0))