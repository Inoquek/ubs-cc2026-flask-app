import logging
from flask import request, jsonify
from routes import app


ATTACK_MINUTES = 10
COOLDOWN_MINUTES = 10

logger = logging.getLogger(__name__)

def solve_case(case):
    intel = case["intel"]
    reserve = case["reserve"]
    fronts = case["fronts"]
    stamina_max = case["stamina"]

    time = 0
    mp = reserve
    stamina = stamina_max
    prev_action_attack = False
    last_front = None

    logger.info({"event": "case_start", "reserve": reserve, "fronts": fronts,
                 "stamina": stamina_max, "intel_len": len(intel)})

    for idx, (front, mp_cost) in enumerate(intel, start=1):
        logger.info({"event": "consider_spell", "seq": idx, "front": front,
                     "mp_cost": mp_cost, "time": time, "mp": mp, "stamina": stamina})

        if mp_cost > mp or stamina == 0:
            logger.info({"event": "cooldown_needed", "seq": idx,
                         "reason": "mp" if mp_cost > mp else "stamina",
                         "time_before": time, "mp_before": mp, "stamina_before": stamina})
            time += 10
            mp = reserve
            stamina = stamina_max
            prev_action_attack = False
            logger.info({"event": "cooldown_done", "seq": idx,
                         "time_after": time, "mp_after": mp, "stamina_after": stamina})

        mp -= mp_cost
        stamina -= 1

        if prev_action_attack and last_front == front:
            logger.info({"event": "attack_extend", "seq": idx, "front": front,
                         "mp_cost": mp_cost, "time": time, "mp_left": mp, "stamina_left": stamina})
        else:
            time += 10
            logger.info({"event": "attack_new", "seq": idx, "front": front,
                         "mp_cost": mp_cost, "time": time, "mp_left": mp, "stamina_left": stamina})

        prev_action_attack = True
        last_front = front

    logger.info({"event": "final_cooldown_before", "time": time, "mp": mp, "stamina": stamina})
    time += 10
    logger.info({"event": "final_cooldown_after", "time": time})

    if "expected" in case and case["expected"] != time:
        logger.info({"event": "test_failed", "expected": case["expected"], "got": time,
                     "reserve": reserve, "fronts": fronts, "stamina": stamina_max})

    return {"time": time}

@app.route("/the-mages-gambit", methods=["POST"])
def gambit():
    if not request.is_json:
        return jsonify({"error": "Expected application/json body"}), 400
    payload = request.get_json()
    if not isinstance(payload, list):
        return jsonify({"error": "Expected a JSON array of cases"}), 400
    
    for case in payload:
        logger.info(case)
    try:
        result = [solve_case(case) for case in payload]
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result), 200
