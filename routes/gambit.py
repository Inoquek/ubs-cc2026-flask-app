import logging
from flask import request, jsonify
from routes import app


ATTACK_MINUTES = 10
COOLDOWN_MINUTES = 10

def solve_case(case):
    intel = case["intel"]
    reserve = case["reserve"]
    fronts = case["fronts"]
    stamina_max = case["stamina"]

    # Basic validation (kept light; challenge promises valid inputs)
    if reserve <= 0 or stamina_max <= 0 or fronts <= 0:
        raise ValueError("reserve, stamina, and fronts must be positive.")
    
    for f, mp in intel:
        if not (1 <= f <= fronts):
            raise ValueError(f"front {f} out of range 1..{fronts}")
        if not (1 <= mp <= reserve):
            raise ValueError(f"mp cost {mp} out of range 1..{reserve}")

    time = 0
    mp = reserve
    stamina = stamina_max

    for front, mp_cost in intel:
        # If the next attack doesn't fit, cooldown until it does
        if mp_cost > mp or stamina == 0 or stamina < front:
            mp = reserve
            stamina = stamina_max
            time += COOLDOWN_MINUTES

        # Now perform the attack
        mp -= mp_cost
        stamina -= 1
        time += ATTACK_MINUTES

    return {"time": time}

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
