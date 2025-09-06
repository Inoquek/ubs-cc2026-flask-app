import logging
from collections import defaultdict, List, tuple
from typing import List, Tuple, Dict, Any, Optional
from flask import request, jsonify
import math
from routes import app
import json
import ast
logger = logging.getLogger(__name__)



def challenge1_calc(data) :
    transformations = data.get("transformations")

    logger.info(transformations)
    encrypted_word = data.get("transformed_encrypted_word")

    words = encrypted_word.split()
    transformations = transformations[::-1]

    def process_words(words, transform):
        if transform == "mirror_words":

            for index, word in enumerate(words):
                words[index] = word[::-1]

        elif transform == "encode_mirror_alphabet":
            
            logger.info("hmmm")
            for index, word in enumerate(words):
                new_word = ""
                for i in range(len(word)):
                    if (word[i] >= 'a' and word[i] <= 'z') :
                        start = 'a'
                        end = 'z'
                    else :
                        start = 'A'
                        end = 'Z'
                    new_word += chr(ord(end) - (ord(word[i]) - ord(start)))

                words[index] = new_word
                logger.info(new_word)

            logger.info(words)
        elif transform == "toggle_case":

            for index, word in enumerate(words):
                new_word = ""
                for i in range(len(word)):
                    if (word[i] >= 'a' and word[i] <= 'z') :
                        start = 'a'
                        startOther = 'A'
                    else :
                        start = 'A'
                        startOther = 'a'
                    new_word += chr(ord(startOther) + (ord(word[i]) - ord(start)))
                words[index] = new_word
        elif transform == "swap_pairs":

            for index, word in enumerate(words):
                new_word = ""
                for i in range(0, len(word) - 1, 2):
                    new_word += word[i + 1]
                    new_word += word[i]
                if len(word) % 2:
                    new_word += word[-1]

                words[index] = new_word
        elif transform == "encode_index_parity":

            for index, word in enumerate(words):
                word_sz = len(word)
                odd_pointer = (word_sz // 2) + word_sz % 2
                even_pointer = 0
                new_word = ""
                for i in range(word_sz):
                    if i % 2:
                        new_word += word[odd_pointer]
                        odd_pointer += 1
                    else : 
                        new_word += word[even_pointer]
                        even_pointer += 1

                words[index] = new_word

        elif transform == "double_consonants":
            for index, word in enumerate(words):
                new_word = ""
                for i in range(len(word)):
                    if word[i] in set(['a', 'o', 'e', 'u', 'i']):
                        new_word += word[i]
                    elif i % 2 == 0:
                        new_word += word[i]
                words[index] = new_word
        
        return words
    
    for transform in transformations:
        unnested = [t.replace(")", "") for t in transform.split('(')]


        logger.info(unnested)
        for t in unnested:
            if t == 'x':
                continue
            words = process_words(words, t)
        

    final_word = " ".join(words)

    return final_word



# --------------------------
# Challenge 2: helpers
# --------------------------
def _parse_coords_list(raw: List[List[Any]]) -> List[Tuple[float, float]]:
    pts = []
    for pair in raw or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            lat = float(pair[0])
            lon = float(pair[1])
            pts.append((lat, lon))
        except Exception:
            continue
    return pts

def _z_clean(points: List[Tuple[float, float]], zmax: float = 2.75) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]:
    """Return (kept_points, removed_outliers) via simple z-score on lat & lon."""
    if not points:
        return [], []
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    mean_lat = sum(lats)/len(lats)
    mean_lon = sum(lons)/len(lons)
    var_lat = sum((x-mean_lat)**2 for x in lats) / max(1, len(lats)-1)
    var_lon = sum((x-mean_lon)**2 for x in lons) / max(1, len(lons)-1)
    std_lat = math.sqrt(var_lat) if var_lat > 0 else 0.0
    std_lon = math.sqrt(var_lon) if var_lon > 0 else 0.0

    kept, removed = [], []
    for (lat, lon) in points:
        zlat = abs((lat-mean_lat)/std_lat) if std_lat > 0 else 0.0
        zlon = abs((lon-mean_lon)/std_lon) if std_lon > 0 else 0.0
        (kept if (zlat <= zmax and zlon <= zmax) else removed).append((lat, lon))
    return kept, removed

def _normalize(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Affine normalize to unit square with aspect preserved, centered to [0,1]x[0,1]."""
    if not points:
        return []
    min_lat = min(p[0] for p in points)
    max_lat = max(p[0] for p in points)
    min_lon = min(p[1] for p in points)
    max_lon = max(p[1] for p in points)
    span_lat = max(max_lat - min_lat, 1e-9)
    span_lon = max(max_lon - min_lon, 1e-9)

    # scale by the larger span so the larger dimension fits to 1
    scale = max(span_lat, span_lon)
    norm = [((p[0]-min_lat)/scale, (p[1]-min_lon)/scale) for p in points]

    # center to [0,1] on the other axis
    nmin_lat = min(x for x, _ in norm)
    nmax_lat = max(x for x, _ in norm)
    nmin_lon = min(y for _, y in norm)
    nmax_lon = max(y for _, y in norm)
    shift_lat = (1.0 - (nmax_lat - nmin_lat)) / 2.0 - nmin_lat
    shift_lon = (1.0 - (nmax_lon - nmin_lon)) / 2.0 - nmin_lon
    return [(x + shift_lat, y + shift_lon) for (x, y) in norm]

def _sample_segment(p0: Tuple[float,float], p1: Tuple[float,float], n: int = 80) -> List[Tuple[float,float]]:
    (x0,y0), (x1,y1) = p0, p1
    if n <= 1:
        return [p0, p1]
    out = []
    for i in range(n):
        t = i/(n-1)
        out.append((x0*(1-t)+x1*t, y0*(1-t)+y1*t))
    return out

def _seven_segment_template_points(active: List[str]) -> List[Tuple[float,float]]:
    """
    Build a point cloud for a 7-seg digit inside [0,1]x[0,1].
    Segment naming:
      A: top, B: top-right, C: bottom-right, D: bottom, E: bottom-left, F: top-left, G: middle
    """
    # Segment endpoints (x,y) with a small inset margin
    m = 0.12  # margin from edges for nicer proportions
    top_y = 1 - m
    bot_y = m
    mid_y = 0.5
    left_x = m
    right_x = 1 - m
    # Slight inner offsets to make verticals distinct from corners
    inner = 0.08
    segs = {
        'A': [(left_x, top_y), (right_x, top_y)],
        'B': [(right_x, top_y), (right_x, mid_y + inner)],
        'C': [(right_x, mid_y - inner), (right_x, bot_y)],
        'D': [(left_x, bot_y), (right_x, bot_y)],
        'E': [(left_x, mid_y - inner), (left_x, bot_y)],
        'F': [(left_x, top_y), (left_x, mid_y + inner)],
        'G': [(left_x + 0.02, mid_y), (right_x - 0.02, mid_y)],
    }
    pts = []
    dens = 90  # points per segment for smoother matching
    for s in active:
        p0, p1 = segs[s]
        pts.extend(_sample_segment(p0, p1, dens))
    return pts

# 7-seg encodings for 0–9
SEG_BY_DIGIT = {
    0: ['A','B','C','D','E','F'],
    1: ['B','C'],
    2: ['A','B','G','E','D'],
    3: ['A','B','G','C','D'],
    4: ['F','G','B','C'],
    5: ['A','F','G','C','D'],
    6: ['A','F','G','E','C','D'],
    7: ['A','B','C'],
    8: ['A','B','C','D','E','F','G'],
    9: ['A','B','C','D','F','G'],
}

TEMPLATE_CACHE: Dict[int, List[Tuple[float,float]]] = {}

def _get_template(d: int) -> List[Tuple[float,float]]:
    if d not in TEMPLATE_CACHE:
        TEMPLATE_CACHE[d] = _seven_segment_template_points(SEG_BY_DIGIT[d])
    return TEMPLATE_CACHE[d]

def _avg_min_distance(A: List[Tuple[float,float]], B: List[Tuple[float,float]]) -> float:
    """Average minimal Euclidean distance from each point in A to nearest in B."""
    if not A or not B:
        return float('inf')
    total = 0.0
    for ax, ay in A:
        best = float('inf')
        # linear search is ok here (N small)
        for bx, by in B:
            dx = ax - bx
            dy = ay - by
            d = dx*dx + dy*dy
            if d < best:
                best = d
        total += math.sqrt(best)
    return total / len(A)

def _try_flips(points: List[Tuple[float,float]]) -> List[List[Tuple[float,float]]]:
    """Try identity + flips (x, y, both) to be robust to axis orientation."""
    idt = points
    flipx = [(1-x, y) for (x,y) in points]
    flipy = [(x, 1-y) for (x,y) in points]
    flipxy = [(1-x, 1-y) for (x,y) in points]
    return [idt, flipx, flipy, flipxy]

def _digit_match(points: List[Tuple[float,float]]) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Return (best_digit, debug_info).
    We compute a symmetric score: avg_min(A->T) + avg_min(T->A) and take the min across flips.
    """
    if not points:
        return None, {"reason": "no_points"}
    P = _normalize(points)
    candidates = list(range(10))
    best_digit = None
    best_score = float('inf')
    best_flip = "identity"

    flip_names = ["identity", "flip_x", "flip_y", "flip_xy"]
    for flip_name, Pvariant in zip(flip_names, _try_flips(P)):
        for d in candidates:
            T = _get_template(d)
            # templates are already normalized in [0,1]
            s1 = _avg_min_distance(Pvariant, T)
            s2 = _avg_min_distance(T, Pvariant)
            score = 0.5*(s1 + s2)
            if score < best_score:
                best_score = score
                best_digit = d
                best_flip = flip_name

    debug = {"score": best_score, "flip": best_flip}
    # Heuristic acceptance threshold — tuned to be forgiving
    if best_score <= 0.12:
        return best_digit, debug
    else:
        return None, {"reason": "ambiguous", **debug}

def _mymaps_csv(points: List[Tuple[float,float]]) -> str:
    lines = ["latitude,longitude"]
    for lat, lon in points:
        lines.append(f"{lat},{lon}")
    return "\n".join(lines)

# --------------------------
# Challenge 2: main
# --------------------------
def challenge2_calc(data):
    """
    Input: list of [lat, lon]
    Output: int digit (0-9) if confidently detected, else None.
    Also logs helpful artifacts and returns extras (added in operation()).
    """
    # Parse
    pts = _parse_coords_list(data)
    logger.info(f"[C2] received {len(pts)} points")

    # Outlier removal
    kept, removed = _z_clean(pts, zmax=2.75)
    logger.info(f"[C2] kept={len(kept)} removed={len(removed)}")

    # Auto-detect digit using 7-seg templates
    digit, dbg = _digit_match(kept)
    logger.info(f"[C2] match={digit} debug={dbg}")

    # Stash extras for the response (so you can quickly visualize)
    extras = {
        "mymaps_csv": _mymaps_csv(kept),
        "removed_as_anomalies": removed,
        "debug": dbg
    }

    # Return both scalar answer and extras
    return digit, extras
def challenge3_calc(data) :
    return "a"
def challenge4_calc(result1, result2, result3) :

    return "a"

@app.route('/operation-safeguard', methods=['POST'])
def operation():
    data = request.get_json()
    logging.info("data sent for evaluation {}".format(data))
    
    challenge1_data = data.get("challenge_one")
    challenge2_data = data.get("challenge_two")
    challenge3_data = data.get("challenge_three")

    result1 = challenge1_calc(challenge1_data)
    result2, c2_extras = challenge2_calc(challenge2_data) = challenge2_calc(challenge2_data)
    result3 = challenge3_calc(challenge3_data)

    result4 = challenge4_calc(result1, result2, result3)

    result = {"challenge_one": result1, "challenge_two": result2, "challenge_three": result3, "challenge_four": result4}
    logging.info("My result :{}".format(result))
    return json.dumps(result)