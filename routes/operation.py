import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
from flask import request, jsonify
import math
from routes import app
import json
import re
import ast
logger = logging.getLogger(__name__)

def challenge1_calc(data):
    """
    Input:
      data = {
        "transformations": [ "encode_mirror_alphabet(x)", "double_consonants(x)", ... ],
        "transformed_encrypted_word": "<ciphertext>"
      }
    Output: plaintext string after applying the inverse of each step in reverse order.
    """
    transformations = data.get("transformations") or []
    s = data.get("transformed_encrypted_word", "")

    # Accept a JSON-stringified list just in case.
    if isinstance(transformations, str):
        try:
            transformations = json.loads(transformations)
        except Exception:
            transformations = [transformations]

    VOWELS = set("aeiouAEIOU")

    # ---------- primitives ----------
    def atbash_char(ch: str) -> str:
        if 'A' <= ch <= 'Z':
            return chr(ord('Z') - (ord(ch) - ord('A')))
        if 'a' <= ch <= 'z':
            return chr(ord('z') - (ord(ch) - ord('a')))
        return ch  # leave non-letters unchanged

    def atbash(s: str) -> str:
        return "".join(atbash_char(c) for c in s)

    def apply_wordwise(s: str, fn):
        # preserve exact spacing: split keeping whitespace tokens
        parts = re.split(r'(\s+)', s)
        return "".join(fn(p) if p and not p.isspace() else p for p in parts)

    def mirror_each_word(s: str) -> str:
        return apply_wordwise(s, lambda w: w[::-1])

    def swap_pairs_word(w: str) -> str:
        ch = list(w)
        for i in range(0, len(ch) - 1, 2):
            ch[i], ch[i+1] = ch[i+1], ch[i]
        return "".join(ch)

    def swap_pairs(s: str) -> str:
        return apply_wordwise(s, swap_pairs_word)

    def decode_index_parity_word(w: str) -> str:
        # inverse of: evens first, then odds
        n = len(w)
        e_cnt = (n + 1) // 2
        ev, od = w[:e_cnt], w[e_cnt:]
        out = []
        ei = oi = 0
        for i in range(n):
            if i % 2 == 0:
                out.append(ev[ei]); ei += 1
            else:
                out.append(od[oi]); oi += 1
        return "".join(out)

    def decode_index_parity(s: str) -> str:
        return apply_wordwise(s, decode_index_parity_word)

    def undouble_consonants_word(w: str) -> str:
        # collapse ONLY true doubled consonants; keep vowels/non-letters as-is
        out = []
        i = 0
        n = len(w)
        while i < n:
            ch = w[i]
            if ch.isalpha() and ch not in VOWELS and i + 1 < n and w[i+1] == ch:
                out.append(ch)
                i += 2
            else:
                out.append(ch)
                i += 1
        return "".join(out)

    def undouble_consonants(s: str) -> str:
        return apply_wordwise(s, undouble_consonants_word)

    # ---------- map of decoders (each is the exact inverse) ----------
    decoders = {
        "mirror_words":          mirror_each_word,      # self-inverse
        "encode_mirror_alphabet": atbash,               # self-inverse (Atbash)
        "toggle_case":           str.swapcase,          # self-inverse
        "swap_pairs":            swap_pairs,            # self-inverse
        "encode_index_parity":   decode_index_parity,   # true inverse
        "double_consonants":     undouble_consonants,   # true inverse
    }

    # For each transformation token (possibly nested like f(g(x))),
    # apply inverse from OUTER to INNER, and process tokens in REVERSE list order.
    # Example encode: step1 = f(g(x)); step2 = h(x)
    # Decode order: inverse(h), then inverse(f), then inverse(g).
    for token in reversed(transformations):
        # Extract names in outer→inner order: "swap_pairs(encode_mirror_alphabet(x))" -> ["swap_pairs","encode_mirror_alphabet"]
        names = re.findall(r'([a-z_]+)\s*\(', token)
        if not names:
            # Handle simple name without "(x)"
            m = re.match(r'^\s*([a-z_]+)\s*$', token)
            if m:
                names = [m.group(1)]

        for name in names:
            dec = decoders.get(name)
            if not dec:
                logger.warning("Unknown transformation '%s' — skipping", name)
                continue
            s = dec(s)

    return s

def challenge2_calc(data) :
   return 'a'


# -----------------------------
# Challenge 3: helpers
# -----------------------------
def _parse_kv_log(line: str) -> dict:
    """Parse 'K: V | K2: V2 | ...' with arbitrary order/casing."""
    out = {}
    for chunk in (line or "").split("|"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        k, v = chunk.split(":", 1)
        out[k.strip().upper()] = v.strip()
    return out

def _rot13(s: str) -> str:
    out = []
    for ch in s:
        if "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + 13) % 26 + 65))
        elif "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + 13) % 26 + 97))
        else:
            out.append(ch)
    return "".join(out)

def _railfence3_decrypt(ct: str) -> str:
    """Rail fence (3 rails) decryption."""
    n = len(ct)
    if n == 0:
        return ct

    # Which row each index belongs to (0,1,2,1,0,...)
    rows = []
    r, step = 0, 1
    for _ in range(n):
        rows.append(r)
        r += step
        if r == 2:
            step = -1
        elif r == 0:
            step = 1

    counts = [rows.count(i) for i in range(3)]

    # Slice ciphertext into row chunks
    idx = 0
    row_chunks = []
    for c in counts:
        row_chunks.append(list(ct[idx:idx + c]))
        idx += c

    # Rebuild plaintext by walking rows pattern
    pos = [0, 0, 0]
    out = []
    for rr in rows:
        out.append(row_chunks[rr][pos[rr]])
        pos[rr] += 1
    return "".join(out)

_ALPHA_UP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def _keyword_alphabet(keyword: str) -> str:
    """Build monoalphabetic cipher alphabet from keyword (dedupe, then A..Z)."""
    seen = set()
    seq = []
    for ch in keyword.upper():
        if ch.isalpha() and ch not in seen:
            seen.add(ch)
            seq.append(ch)
    for ch in _ALPHA_UP:
        if ch not in seen:
            seen.add(ch)
            seq.append(ch)
    return "".join(seq)

def _keyword_decrypt(ct: str, keyword: str = "SHADOW") -> str:
    """Simple substitution decrypt where cipher alphabet = keyword alphabet."""
    c_alph = _keyword_alphabet(keyword)  # index maps PLAIN->CIPHER
    out = []
    for ch in ct:
        if ch.isalpha():
            if ch.isupper():
                i = c_alph.find(ch)
                out.append(_ALPHA_UP[i] if i >= 0 else ch)
            else:
                i = c_alph.find(ch.upper())
                out.append(_ALPHA_UP[i].lower() if i >= 0 else ch)
        else:
            out.append(ch)
    return "".join(out)

def _polybius_decrypt(ct: str) -> str:
    """
    Polybius 5x5 (I/J combined). Accepts digits with any separators (e.g., '11 21 31', '112131').
    Non-digits are ignored. Produces UPPERCASE letters with J→I.
    """
    grid = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # No J
    digits = re.findall(r"\d", ct)
    if len(digits) % 2 == 1:
        digits = digits[:-1]  # drop trailing odd digit, if any
    out = []
    for i in range(0, len(digits), 2):
        r = int(digits[i])
        c = int(digits[i+1])
        if 1 <= r <= 5 and 1 <= c <= 5:
            out.append(grid[(r-1)*5 + (c-1)])
    return "".join(out)

# -----------------------------
# Challenge 3: main function
# -----------------------------
def challenge3_calc(data: str) -> str:
    """
    data: the full log line, e.g.
      "PRIORITY: HIGH | ... | CIPHER_TYPE: ROTATION_CIPHER | ... | ENCRYPTED_PAYLOAD: SVERJNYY | ..."
    returns: decrypted payload as a STRING (grader requires string)
    """
    kv = _parse_kv_log(data or "")
    ctype = (kv.get("CIPHER_TYPE") or kv.get("CIPHER") or "").upper()
    payload = kv.get("ENCRYPTED_PAYLOAD") or ""

    if not payload:
        logger.error("[C3] No ENCRYPTED_PAYLOAD in log")
        return ""

    logger.info(f"[C3] cipher={ctype} payload_len={len(payload)}")

    # Route to the right cipher
    if "ROTATION" in ctype or "ROT" in ctype:
        res = _rot13(payload)
    elif "RAILFENCE" in ctype:
        res = _railfence3_decrypt(payload)
    elif "KEYWORD" in ctype:
        res = _keyword_decrypt(payload, "SHADOW")
    elif "POLYBIUS" in ctype:
        res = _polybius_decrypt(payload)
    else:
        # Fallback: try ROT13 (common in samples)
        res = _rot13(payload)

    logger.info(f"[C3] decrypted='{res}'")
    # Ensure string return (grader requirement)
    return str(res)


def challenge4_calc(result1, result2, result3) :

    return ""

@app.route('/operation-safeguard', methods=['POST'])
def operation():
    data = request.get_json()
    logging.info("data sent for evaluation {}".format(data))
    
    challenge1_data = data.get("challenge_one")
    challenge2_data = data.get("challenge_two")
    challenge3_data = data.get("challenge_three")
    
    result1 = challenge1_calc(challenge1_data)
    result2 = challenge2_calc(challenge2_data)

    result3 = challenge3_calc(challenge3_data)
    result4 = challenge4_calc(result1, result2, result3)
    
    # IMPORTANT: the grader wants strings for these four keys
    result = {
        "challenge_one":   result1,
        "challenge_two":   result2,
        "challenge_three": result3,
        "challenge_four":  result4,
    }
    logging.info("My result :{}".format(result))
    return json.dumps(result)