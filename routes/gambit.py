import logging
from flask import request, jsonify
from routes import app


ATTACK_MINUTES = 10
COOLDOWN_MINUTES = 10

logging.basicConfig(level=logging.INFO)
FAIL_LOGGER = logging.getLogger(__name__)
FAIL_LOGGER.setLevel(logging.INFO)   # <= now logs at INFO 

def _log_failed_test(case, got, expected):
    FAIL_LOGGER.info({
        "event": "test_failed",
        "intel": case.get("intel"),
        "reserve": case.get("reserve"),
        "fronts": case.get("fronts"),
        "stamina": case.get("stamina"),
        "expected": expected,
        "got": got,
    })

def solve_case(case):
    intel = case["intel"]
    reserve = case["reserve"]
    fronts = case["fronts"]
    stamina_max = case["stamina"]

    # Lightweight validation
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

    # Track whether the previous action was an ATTACK (not a cooldown)
    # and on which front. Only then can we apply the 0-minute "extend AOE".
    prev_action_attack = False
    last_front = None

    for front, mp_cost in intel:
        # If we can't cast the next spell, cooldown first.
        if mp_cost > mp or stamina == 0:
            time += COOLDOWN_MINUTES
            mp = reserve
            stamina = stamina_max
            prev_action_attack = False  # breaks the "extend AOE" chain

        # Perform the attack
        mp -= mp_cost
        stamina -= 1

        # Time cost: 0 if immediately after an attack on the same front;
        # otherwise 10 minutes (retargeting / first attack in a chain).
        if prev_action_attack and last_front == front:
            # extend AOE — no extra time
            pass
        else:
            time += ATTACK_MINUTES

        prev_action_attack = True
        last_front = front

    # Must be in cooldown state at the end (to join expedition ready)
    time += COOLDOWN_MINUTES
    prev_action_attack = False  # for completeness

    # Optional: compare with provided expected, log mismatch
    if "expected" in case and case["expected"] != time:
        _log_failed_test(case, got=time, expected=case["expected"])

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
