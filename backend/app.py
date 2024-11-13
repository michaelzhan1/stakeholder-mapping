from flask import Flask, request, abort
from flask_cors import CORS
from dotenv import load_dotenv
from train import run
import numpy as np
import os

load_dotenv()

app = Flask(__name__)
CORS(app, origins=[os.getenv('FRONTEND_URL')])

@app.route('/api/rl-endpoint', methods=['POST'])
def run_rl():
    if request.content_type != 'text/plain':
        abort(400, description='Content-Type must be text/plain')
    
    data = request.get_data(as_text=True).replace("\\n", "\n")
    formatted_data = np.array([list(map(int, line.split(','))) for line in data.split('\n')])

    output = run(formatted_data)
    return output
