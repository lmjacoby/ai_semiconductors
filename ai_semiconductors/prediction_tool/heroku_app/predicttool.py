from flask import Flask

app = Flask(__name__)

@app.route("/")

def home():
    return '<h1>This is the prediction tool app</h1>'
