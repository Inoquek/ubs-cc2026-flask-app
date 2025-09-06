from flask import Flask
app = Flask(__name__)
import routes.square 
import routes.agent
import routes.trader
import routes.trivia
import routes.gambit
import routes.duolingo