from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import warnings

# Ignore specific warnings globally
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from ecommbot.retrieval_generation import generation
from ecommbot.ingest import ingestdata

app = Flask(__name__)
load_dotenv()

# Load vector store and generation chain
vstore = ingestdata("done")
chain = generation(vstore)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    result = chain.invoke(msg)

    # Extract only the final answer, assuming it's a plain string
    return str(result)

if __name__ == '__main__':
    app.run(debug=True)
