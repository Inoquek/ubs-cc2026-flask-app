# routes/trivia.py
import os
from flask import jsonify, request
from routes import app

# Optional: set TRIVIA_ANSWERS in Railway (comma-separated ints)
# e.g. "1,3,2,2,4,4,3,1,2"

# hz naschest 6,7,
# definitely correct 3,1,2,2,3, ... , 4 , 5, 4,3,3, , ,  
_RAW = os.getenv("TRIVIA_ANSWERS", "3,1,2,2,3,4,4,5,4,3,    2, 2,      -1, -1, 2    , 1, 1,2, -1, 1,     -1,2,-1,4,2")

# new changes (not confident in them)
# Q.24 - 4 IS CORRECT
# Q.19 - trying 2(deleting this one)
# Q.11 - not 1,3
# Q.12 - not 1
# Q.21 - 4 (deleting)

# TRYING Q.11 and Q.12

# -1 if dk the answer yet
# everything else is correct so far

try:
    ANSWERS = [int(x.strip()) for x in _RAW.split(",") if x.strip()]
except ValueError:
    # Fallback to an empty list if env var is malformed
    ANSWERS = []

@app.route("/trivia", methods = ["GET"])
def trivia():
    return jsonify({"answers": ANSWERS})
