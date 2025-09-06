import logging
from collections import defaultdict
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