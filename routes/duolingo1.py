import os
import re
import logging
from typing import List, Tuple, Dict, Callable
from flask import request, jsonify
from routes import app

from collections import defaultdict

        
# ---------------------------
# Minimal logger: only for targeted debug lines
# ---------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_h)


def _log_only_when_wrong(part: str, challenge, unsorted_list, annotated):
    """
    annotated = list of tuples: (value, tie_rank, idx, original)
    Logs nothing if everything looks correct.
    Logs a single compact ERROR line if any check fails.
    """
    issues = []

    # A) Numeric order must be non-decreasing
    for i in range(1, len(annotated)):
        v_prev, _, _, s_prev = annotated[i-1]
        v_curr, _, _, s_curr = annotated[i]
        if v_curr < v_prev:
            issues.append(f"ORDER i={i} prev='{s_prev}'({v_prev}) > curr='{s_curr}'({v_curr})")
            break  # one is enough to flag

    # B) Within equal numeric values, tie ranks must follow your policy order
    if not issues:
        by_value = defaultdict(list)  # value -> [(tie, idx, s), ...] in final order
        for (v, tie, idx, s) in annotated:
            by_value[v].append((tie, idx, s))
        for v, group in by_value.items():
            expected = sorted(group, key=lambda x: (x[0], x[1]))  # (tie, idx)
            if group != expected:
                got = [s for (_, _, s) in group]
                exp = [s for (_, _, s) in expected]
                issues.append(f"TIE value={v} got={got} expected={exp}")
                break

    if issues:
        # Keep log tiny: first 5 inputs & first issue only
        logger.error(
            "WRONG part=%s challenge=%s n=%d sample_in=%s issue=%s",
            part, str(challenge), len(unsorted_list), unsorted_list[:5], issues[0]
        )
        
# ---------------------------
# Utilities: Roman numerals
# ---------------------------
_ROMAN_MAP = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
_VALID_ROMAN = re.compile(r'^[IVXLCDM]+$')

def roman_to_int(s: str) -> int:
    if not _VALID_ROMAN.match(s):
        raise ValueError("invalid roman characters")
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_MAP[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    if total < 1 or total > 3999:
        raise ValueError("roman out of range")
    return total

# ---------------------------
# Utilities: Arabic numerals
# ---------------------------
_DIGITS_ONLY = re.compile(r'^\d+$')

def arabic_to_int(s: str) -> int:
    if not _DIGITS_ONLY.match(s):
        raise ValueError("invalid arabic digits")
    return int(s)

# ---------------------------
# English words (kept for Part TWO completeness)
# ---------------------------
_EN_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
}
_EN_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19
}
_EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}
_EN_SCALES = {
    "billion": 1_000_000_000,
    "million": 1_000_000,
    "thousand": 1_000,
    "hundred": 100
}
_EN_TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)?")

def english_to_int(s: str) -> int:
    s = s.lower().strip()
    tokens = []
    for part in s.replace("-", " ").split():
        if not _EN_TOKEN.fullmatch(part):
            raise ValueError("invalid english token")
        tokens.append(part)
    if not tokens:
        raise ValueError("empty english")
    total = 0
    chunk = 0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in _EN_UNITS:
            chunk += _EN_UNITS[t]
        elif t in _EN_TEENS:
            chunk += _EN_TEENS[t]
        elif t in _EN_TENS:
            val = _EN_TENS[t]
            if i + 1 < len(tokens) and tokens[i+1] in _EN_UNITS:
                val += _EN_UNITS[tokens[i+1]]
                i += 1
            chunk += val
        elif t in _EN_SCALES:
            scale = _EN_SCALES[t]
            if scale == 100:
                if chunk == 0:
                    chunk = 1
                chunk *= 100
            else:
                total += chunk * scale
                chunk = 0
        elif t == "and":
            pass
        else:
            raise ValueError(f"unknown english word: {t}")
        i += 1
    return total + chunk

# ---------------------------
# German
# ---------------------------
def _normalize_de(s: str) -> str:
    s = s.lower()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return s

_DE_UNITS = {
    "null": 0, "ein": 1, "eins": 1, "eine": 1, "einen": 1, "einem": 1, "einer": 1,
    "zwei": 2, "drei": 3, "vier": 4, "fuenf": 5, "funf": 5, "sechs": 6,
    "sieben": 7, "acht": 8, "neun": 9
}
_DE_TEENS = {
    "zehn": 10, "elf": 11, "zwoelf": 12,
    "dreizehn": 13, "vierzehn": 14, "fuenfzehn": 15, "funfzehn": 15,
    "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19
}
_DE_TENS = {
    "zwanzig": 20, "dreissig": 30,
    "vierzig": 40, "fuenfzig": 50, "funfzig": 50,
    "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90
}
_DE_KEYWORDS = ("und","tausend","hundert","zig","zehn","elf","zwoelf","zwanzig",
                "dreissig","vierzig","fuenfzig","sechzig","siebzig","achtzig","neunzig",
                "null","eins","ein","zwei","drei","vier","fuenf","sechs","sieben","acht","neun")

def _parse_de_under_100(s: str) -> int:
    if s in _DE_UNITS: return _DE_UNITS[s]
    if s in _DE_TEENS: return _DE_TEENS[s]
    if s in _DE_TENS:  return _DE_TENS[s]
    if "und" in s:
        left, right = s.split("und", 1)
        tens = _DE_TENS.get(right)
        unit = _DE_UNITS.get(left)
        if tens is not None and unit is not None:
            return tens + unit
    raise ValueError(f"cannot parse german <100: {s}")

def german_to_int(s_input: str) -> int:
    s = _normalize_de(s_input).replace("-", "").replace(" ", "")
    if not s:
        raise ValueError("empty german")
    if "tausend" in s:
        left, right = s.split("tausend", 1)
        left_val = 1 if left == "" else german_to_int(left)
        return left_val * 1000 + (german_to_int(right) if right else 0)
    if "hundert" in s:
        left, right = s.split("hundert", 1)
        left_val = 1 if left in ("", "ein", "eins") else german_to_int(left)
        return left_val * 100 + (german_to_int(right) if right else 0)
    return _parse_de_under_100(s)

# ---------------------------
# Chinese
# ---------------------------
CN_DIGITS = {'零':0,'〇':0,'○':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'兩':2,'两':2}
CN_UNITS  = {'十':10,'百':100,'千':1000}
CN_BIG_UNITS = {'萬':10_000,'亿':100_000_000,'億':100_000_000,'万':10_000}

def _parse_cn_under_10000(s: str) -> int:
    if not s: return 0
    total = 0; num = 0; unit_hit = False
    if s[0] == '十':
        total += 10; s = s[1:]
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in CN_DIGITS:
            num = CN_DIGITS[ch]; i += 1
            if i < len(s) and s[i] in CN_UNITS:
                total += num * CN_UNITS[s[i]]; unit_hit = True; i += 1; num = 0
        elif ch in CN_UNITS:
            mul = CN_UNITS[ch]
            total += (num if num != 0 else 1) * mul
            unit_hit = True; num = 0; i += 1
        elif ch in ('零','〇','○'):
            i += 1
        else:
            raise ValueError(f"invalid chinese char: {ch}")
    if num != 0 or not unit_hit:
        total += num
    return total

def chinese_to_int(s: str) -> int:
    def split_by_big(txt: str, big: str):
        if big in txt:
            i = txt.index(big); return txt[:i], txt[i+1:]
        return "", txt
    left_yi = ""; rest = s
    for big in ("億","亿"):
        if big in rest:
            left_yi, rest = split_by_big(rest, big); break
    val = 0
    if left_yi != "":
        val += _parse_cn_under_10000(left_yi) * 100_000_000
    left_wan = ""
    for big in ("萬","万"):
        if big in rest:
            left_wan, rest = split_by_big(rest, big); break
    if left_wan != "":
        val += _parse_cn_under_10000(left_wan) * 10_000
    val += _parse_cn_under_10000(rest)
    return val

def is_traditional_cn(s: str) -> bool:
    if any(ch in s for ch in ("萬","億")): return True
    if any(ch in s for ch in ("万","亿")): return False
    return True

# ---------------------------
# Language detection
# ---------------------------
LANG_ROMAN = "ROMAN"; LANG_EN = "EN"; LANG_ZH_TRAD = "ZH_T"; LANG_ZH_SIMP = "ZH_S"; LANG_DE = "DE"; LANG_AR = "AR"
TIE_ORDER = {LANG_ROMAN:0, LANG_EN:1, LANG_ZH_TRAD:2, LANG_ZH_SIMP:3, LANG_DE:4, LANG_AR:5}

def detect_language(s: str) -> str:
    s_stripped = s.strip()
    if _DIGITS_ONLY.match(s_stripped): return LANG_AR
    if s_stripped.isupper() and _VALID_ROMAN.match(s_stripped): return LANG_ROMAN
    if any(ch in s_stripped for ch in list(CN_DIGITS.keys()) + list(CN_UNITS.keys()) + list(CN_BIG_UNITS.keys())):
        return LANG_ZH_TRAD if is_traditional_cn(s_stripped) else LANG_ZH_SIMP
    low = s_stripped.lower()
    if any(w in low for w in ["hundred","thousand","million","billion","and",
                              "one","two","three","four","five","six","seven","eight","nine",
                              "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen",
                              "twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]):
        return LANG_EN
    if any(k in _normalize_de(low) for k in _DE_KEYWORDS): return LANG_DE
    try:
        english_to_int(s_stripped); return LANG_EN
    except Exception:
        pass
    try:
        german_to_int(s_stripped); return LANG_DE
    except Exception:
        pass
    raise ValueError("Unrecognized number language/format")

PARSERS: Dict[str, Callable[[str], int]] = {
    LANG_ROMAN: roman_to_int,
    LANG_EN: english_to_int,
    LANG_ZH_TRAD: chinese_to_int,
    LANG_ZH_SIMP: chinese_to_int,
    LANG_DE: german_to_int,
    LANG_AR: arabic_to_int,
}

# ---------------------------
# Endpoint
# ---------------------------
@app.route("/duolingo-sort", methods=["POST"])
def duolingo1():
    """
    Input JSON:
      { "part": "ONE" | "TWO", "challenge": <int>, "challengeInput": { "unsortedList": [<str>, ...] } }
    Output JSON:
      { "sortedList": [<str>, ...] }
    """
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid request root"}), 400

    part = payload.get("part")
    challenge_input = payload.get("challengeInput", {})
    if part not in ("ONE", "TWO"):
        return jsonify({"error": "part must be 'ONE' or 'TWO'"}), 400
    if not isinstance(challenge_input, dict):
        return jsonify({"error": "challengeInput must be an object"}), 400
    unsorted_list = challenge_input.get("unsortedList")
    if not isinstance(unsorted_list, list) or not all(isinstance(x, str) for x in unsorted_list):
        return jsonify({"error": "unsortedList must be a list of strings"}), 400

    try:
        if part == "ONE":
            values: List[int] = []
            for s in unsorted_list:
                s2 = s.strip()
                if _DIGITS_ONLY.match(s2):
                    values.append(arabic_to_int(s2))
                else:
                    values.append(roman_to_int(s2))
            values.sort()
            return jsonify({"sortedList": [str(v) for v in values]})

        else:  # part == "TWO"
            logger.info(f"Given input = {unsorted_list}")
            
            annotated: List[Tuple[int, int, int, str]] = []
            for idx, s in enumerate(unsorted_list):
                lang = detect_language(s)
                val = PARSERS[lang](s.strip())
                
                tie_rank = TIE_ORDER[lang]
                annotated.append((val, tie_rank, idx, s))

            annotated.sort(key=lambda t: (t[0], t[1], t[2]))
            sorted_list = [t[3] for t in annotated]
            # log the final sorted list
            _log_only_when_wrong(part, payload.get("challenge"), unsorted_list, annotated)

            return jsonify({"sortedList": sorted_list})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception:
        return jsonify({"error": "Internal error"}), 500

# Local run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
