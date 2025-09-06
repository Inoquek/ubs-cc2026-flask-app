# routes/trivia.py
import os
from flask import jsonify, request
from routes import app

# Optional: set TRIVIA_ANSWERS in Railway (comma-separated ints)
# e.g. "1,3,2,2,4,4,3,1,2"

# hz naschest 6,7,
# definitely correct 3,1,2,2,3, ... , 4 , 5, 4, 
_RAW = os.getenv("TRIVIA_ANSWERS", "3,1,2,2,3,1,4,5,4")
# _RAW = os.getenv("TRIVIA_ANSWERS", "3,1,2,2,3,3,4,5,4,  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1")  # default placeholder
#  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
#  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
try:
    ANSWERS = [int(x.strip()) for x in _RAW.split(",") if x.strip()]
except ValueError:
    # Fallback to an empty list if env var is malformed
    ANSWERS = []

@app.route("/trivia", methods = ["GET"])
def trivia():
    return jsonify({"answers": ANSWERS})
