import logging
from flask import request, jsonify
from routes import app

logger = logging.getLogger(__name__)


VIP_POINTS = 100
CC_PRIORITY_POINTS = 50

# Squared distance thresholds to avoid sqrt
D2_30 = 2 * 2   # distance ≤ 2 → +30
D2_20 = 3 * 3   # 2 < distance ≤ 3 → +20

def latency_points_by_d2(d2: int) -> int:
    if d2 <= D2_30:
        return 30
    if d2 <= D2_20:
        return 20
    return 0


class Sol:
    """
    Compute, for each customer, the concert with the highest TOTAL POINTS:
      total = VIP(0/100) + CC priority(0/50 for that concert) + latency points(0/20/30)
    Tie handling: if totals tie, pick the first concert in the input order.
    """

    def __init__(self, customers, concerts, priority_map):
        self.customers = customers
        self.priority_map = priority_map

        # Pre-parse concerts into a stable list (input order preserved)
        self.concerts_parsed = []
        for c in concerts:
            name = c["name"]
            bx, by = c["booking_center_location"]
            self.concerts_parsed.append((name, int(bx), int(by)))

    def solve(self):
        result = {}

        for cust in self.customers:
            cname = cust["name"]
            vip = bool(cust["vip_status"])
            cx, cy = map(int, cust["location"])
            cc = cust["credit_card"]

            base = VIP_POINTS if vip else 0
            preferred = self.priority_map.get(cc)

            best_total = None
            best_concert = None  # first max wins (stable)

            for concert_name, bx, by in self.concerts_parsed:
                dx, dy = cx - bx, cy - by
                d2 = dx * dx + dy * dy

                lp = latency_points_by_d2(d2)
                cc_pts = CC_PRIORITY_POINTS if preferred == concert_name else 0
                total = base + cc_pts + lp

                if (best_total is None) or (total > best_total):
                    best_total = total
                    best_concert = concert_name
                # if total == best_total: do nothing → keep the earlier (stable) one

            result[cname] = best_concert
        return result


@app.route('/ticketing-agent', methods=['POST'])
def agent():
    data = request.get_json(silent=True) or {}
    # logger.info("data sent for evaluation %s", data)

    # Extract required fields
    try:
        customers = data['customers']
        concerts = data['concerts']
        priority = data['priority']
    except KeyError:
        return jsonify(error="Payload must include 'customers', 'concerts', and 'priority'"), 400

    try:
        solver = Sol(customers, concerts, priority)
        solution = solver.solve()
        return jsonify(solution), 200
    except Exception as e:
        # logger.exception("Failed to compute solution")
        return jsonify(error=str(e)), 400
