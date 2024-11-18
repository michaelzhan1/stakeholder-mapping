from flask import Flask, request, abort, send_file
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
    
    data = request.get_data(as_text=True).replace("\\n", "\n").strip()
    formatted_data = np.array([list(map(int, line.split(','))) for line in data.split('\n')])

    output = run(formatted_data, save=True)
    return output

@app.route('/api/rl-endpoint/gif', methods=['POST'])
def get_rl_gif():
    fname = 'stakeholder_network.gif'

    if not os.path.exists(fname):
        abort(404, description='GIF not found')
    
    return send_file(fname, mimetype='image/gif')
