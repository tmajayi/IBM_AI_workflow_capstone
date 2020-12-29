import argparse
from flask import Flask, jsonify, request
from flask import render_template, send_from_directory
import joblib
import socket
import json
import numpy as np
import pandas as pd
import os
import re, time

## import model specific functions and variables
from model_1 import model_train, model_load, model_predict, model_eval
from model_1 import MODEL_VERSION
from logger import update_train_log, update_predict_log

app = Flask(__name__)

@app.route("/")
def landing():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/running', methods=['POST'])
def running():
    return render_template('running.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    """
    basic predict function for the API """
   
     ## input checking

    if 'country' not in request.json:
        print("ERROR API (predict): received request, but no 'country' found within")
        return jsonify([])
        
    if 'date' not in request.json:
        print("ERROR API (predict): received request, but no 'date' found within")
        return jsonify([])

    print("... training model")
    query = request.json
    start_time = time.time()
    model = model_train(query)
    runtime = time.time() - start_time
    print("... training complete")
    
    #update train log
    update_train_log(query['country'], query['date'], runtime, MODEL_VERSION)

    return(jsonify(True))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    basic predict function for the API
    """

    ## input checking
    #if not request.json:
    #   print("ERROR: API (predict): did not receive request data")
    #   return jsonify([])

    if 'country' not in request.json:
        print("ERROR API (predict): received request, but no 'country' found within")
        return jsonify([])
        
    if 'date' not in request.json:
        print("ERROR API (predict): received request, but no 'date' found within")
        return jsonify([])
    
    print("please wait...")
    query = request.json
    start_time = time.time()
    forecast = model_predict(query)
    #update prediction log
    runtime = time.time() - start_time
    
    eval_results = model_eval()
    update_predict_log(float(sum(forecast)), query['country'], query['date'],eval_results, MODEL_VERSION, runtime)
    print('All done!')
    return({'forecast':int(sum(forecast))})


@app.route('/logs/<filename>', methods=['GET','POST'])
def logs(filename):
    """
    API endpoint to get logs
    """
    log_dir = os.path.join('logs',filename)
    if not os.path.exists(log_dir):
        raise Exception('Cannot find the specified log file')
    log = pd.read_csv(log_dir)
    content = list(log.values[0][2:])
    return json.dumps(content)
    ## YOUR CODE HERE
    ## get the log directory (log_dir) and raise exceptions if you cannot find the log files.
    ## Then return the content of the logs using the flask function send_from-directory()


port = 8882

if __name__ == '__main__':


    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=port)
    else:
        app.run(host='0.0.0.0', threaded=True ,port=port)

