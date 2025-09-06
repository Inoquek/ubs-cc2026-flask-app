# routes/investigate.py
import logging
from collections import defaultdict
from flask import request, jsonify
from routes import app
import json
import numpy as np
logger = logging.getLogger(__name__)


# def calc1(data):
#     ratios = data.get("ratios")
#     goods = data.get("goods")

#     n = len(goods)
#     connectivity_list = np.array([[] for _ in range(n)])

#     for ratio in ratios:
#         connectivity_list[int(ratio[0])].append([int(ratio[1]), np.float64(ratio[2])])

    
#     num_mask = 2 ** n
#     dp = np.array([[[0.0 for _ in range(num_mask)] for ___ in range(n)] for __ in range(n)]).astype(np.float64)
#     pr = np.array([[[1 for _ in range(num_mask)] for ___ in range(n)] for __ in range(n)])

#     mx = (0, 0, 0)

#     logger.info("Start!")

#     for i in range(n):
#         dp[i][i][2 ** i] = 1.0
#         for mask in range(0, num_mask):
#             if (mask & int(2 ** i)) == 0:
#                 continue

#             for prev_end in range(n):
#                 if (mask & int(2 ** prev_end)) == 0:
#                     continue
#                 for to_edge in connectivity_list[prev_end]:
#                     to = to_edge[0]
#                     w = to_edge[1]
#                     if to == i:
#                         if dp[i][prev_end][mask] * to_edge[1] > dp[i][i][mask]:
#                             pr[i][i][mask] = prev_end
#                             dp[i][i][mask] = dp[i][prev_end][mask] * to_edge[1]
#                     if (mask & int(2 ** to)) == 0:
#                         continue

#                     nmask = (mask ^ int(2 ** to))
#                     if dp[i][prev_end][nmask] * w > dp[i][to][mask]:
#                         dp[i][to][mask] = dp[i][prev_end][nmask] * w
#                         pr[i][to][mask] = prev_end


#             if dp[i][i][mask] > mx[0]:
#                 mx = (dp[i][i][mask], i, mask)


#     logger.info("DP computed")
#     gain, start, mask = mx
#     v = pr[start][start][mask]
#     path = [goods[start], goods[v]]

#     logger.info([gain, start, mask, v])
#     while v != start:
#         new_mask = (mask ^ int(2 ** v))
#         logger.info(dp[start][v][mask])
#         v = pr[start][v][mask]
#         mask = new_mask
#         path.append(goods[v])
#         logger.info([new_mask, v])

#     logger.info("Finished???")
#     logger.info(path)
#     rev_path = path[::-1]

#     logger.info("Ready to Return")
#     return {"path": rev_path, "gain": float((gain - 1.0) * 100.0)}

# def calc1(data):
#     ratios = data.get("ratios")
#     goods = data.get("goods")

#     n = len(goods)
#     # Use list of lists instead of numpy array for connectivity
#     connectivity_list = [[] for _ in range(n)]

#     for ratio in ratios:
#         connectivity_list[int(ratio[0])].append([int(ratio[1]), np.float64(ratio[2])])

#     num_mask = 2 ** n
#     mx = 1.0
#     path = [goods[0]]
#     logger.info("Start!")

#     for i in range(n):
        
#         dp = np.zeros((n, num_mask), dtype=np.float64)
#         pr = np.ones((n, num_mask), dtype=np.int32)
#         dp[i, 2**i] = 1.0
#         for mask in range(1, num_mask):
#             if (mask & (2**i)) == 0:
#                 continue

#             for prev_end in range(n):
#                 if (mask & (2**prev_end)) == 0:
#                     continue
                    
#                 for to_edge in connectivity_list[prev_end]:
#                     to = to_edge[0]
#                     w = to_edge[1]
                    
#                     if to == i:
#                         # Path that ends at i
#                         if dp[prev_end, mask] * w > dp[i, mask]:
#                             pr[i, mask] = prev_end
#                             dp[i, mask] = dp[prev_end, mask] * w
                    
#                     # Skip if 'to' is not in the mask
#                     if (mask & (2**to)) == 0:
#                         continue
                        
#                     # Remove 'to' from mask to get previous state
#                     nmask = mask & ~(2**to)
#                     if dp[prev_end, nmask] * w > dp[to, mask]:
#                         dp[to, mask] = dp[prev_end, nmask] * w
#                         pr[to, mask] = prev_end

#             if dp[i, mask] > mx:
#                 start = i
#                 mx = dp[i, mask]
#                 v = pr[start, mask]
#                 path = [goods[start], goods[v]]
                
#                 logger.info([mx, start, mask, v])
                
#                 while v != start:
#                     current_mask = mask
#                     current = v
#                     v = pr[current, current_mask]
#                     # Only remove the current vertex after we've used it for lookup
#                     mask = mask & ~(2**current)
#                     path.append(goods[v])

#     logger.info("DP computed")
#     rev_path = path[::-1]

#     logger.info("Ready to Return")
#     return {"path": rev_path, "gain": float((mx - 1.0) * 100.0)}

def calc1(data):
    """
    Part I ("The Whispering Loop"): return the best profitable triangle
    (i -> j -> k -> i). If none exists, return empty/0 gain.
    """
    ratios = data.get("ratios") or []
    goods = data.get("goods") or []
    n = len(goods)

    # Dense rate lookup (0.0 means no edge)
    rate = [[0.0] * n for _ in range(n)]
    for u, v, r in ratios:
        ui = int(u); vi = int(v)
        if 0 <= ui < n and 0 <= vi < n:
            rate[ui][vi] = float(r)

    best_prod = 1.0
    best_triangle = None  # (i, j, k) means i->j->k->i

    # Enumerate all distinct triples i, j, k
    for i in range(n):
        for j in range(n):
            if j == i or rate[i][j] <= 0.0:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                if rate[j][k] <= 0.0 or rate[k][i] <= 0.0:
                    continue
                prod = rate[i][j] * rate[j][k] * rate[k][i]
                if prod > best_prod:
                    best_prod = prod
                    best_triangle = (i, j, k)

    if not best_triangle or best_prod <= 1.0:
        return {"path": [], "gain": 0.0}

    i, j, k = best_triangle
    path = [goods[i], goods[j], goods[k], goods[i]]
    gain = (best_prod - 1.0) * 100.0
    return {"path": path, "gain": float(gain)}

def calc2(data):
    """
    Find the maximum-gain arbitrage (simple cycle with max product of rates).
    Returns {"path": [names...], "gain": (product-1)*100}.
    """
    ratios = data.get("ratios") or []
    goods = data.get("goods") or []
    n = len(goods)

    # Build adjacency list: u -> list[(v, rate)]
    adj = [[] for _ in range(n)]
    for u, v, r in ratios:
        ui = int(u)
        vi = int(v)
        rate = float(r)
        if ui < 0 or ui >= n or vi < 0 or vi >= n:
            continue
        if rate <= 0.0:
            continue
        adj[ui].append((vi, rate))

    best_prod = 1.0
    best_cycle_nodes = None  # list of node indices including start at both ends

    # To reduce duplicate cycle exploration, only allow cycles that
    # "start" at the smallest index present in the cycle.
    # Implementation: during DFS from 'start', only take the first edge to 'next'
    # if next >= start. Subsequent steps are unrestricted (except simple-path rule).
    def dfs(start, node, product, path, visited, first_step):
        nonlocal best_prod, best_cycle_nodes

        for (to, rate) in adj[node]:
            if to == start and len(path) >= 2:
                total = product * rate
                if total > best_prod:
                    best_prod = total
                    best_cycle_nodes = path + [start]
            elif to not in visited and len(path) < n:
                # simple path constraint
                if first_step and to < start:
                    # enforce canonical orientation to avoid duplicates
                    continue
                visited.add(to)
                dfs(start, to, product * rate, path + [to], visited, False)
                visited.remove(to)

    for start in range(n):
        if not adj[start]:
            continue
        visited = set([start])
        # first step: enforce to >= start for deduplication
        for (to, rate) in adj[start]:
            if to < start:
                continue
            visited.add(to)
            dfs(start, to, rate, [start, to], visited, False)
            visited.remove(to)

    if best_cycle_nodes is None:
        # No arbitrage cycle found; return "no-op" path and zero gain
        return {"path": [], "gain": 0.0}

    # Convert node indices to goods names
    path_names = [goods[i] for i in best_cycle_nodes]
    gain = (best_prod - 1.0) * 100.0
    return {"path": path_names, "gain": float(gain)}
    
@app.route("/The-Ink-Archive", methods=["POST"])
def ink():
    """
    Challenge server POSTs a JSON array with two items:
      - data[0]: find any surplus cycle (your existing calc1 logic)
      - data[1]: find the maximum gain cycle (new calc2)
    Respond with:
      [
        {"path": [...], "gain": ...},
        {"path": [...], "gain": ...}
      ]
    """
    data = request.get_json()
    logger.info("Received testcases: %s", json.dumps(data, indent=2))
    
    # logger.info("data sent for evaluation %s", data)

    # Defensive checks
    if not isinstance(data, list) or len(data) < 2:
        return jsonify({"error": "Input must be a list with two challenge items"}), 400

    res1 = calc1(data[0])
    res2 = calc2(data[1])

    result = [res1, res2]
    # logger.info("investigate result: %s", result)
    logger.info("investigate result: %s", result)
    return jsonify(result)