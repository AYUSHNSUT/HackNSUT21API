# app.py
from flask import Flask, request, jsonify
app = Flask(__name__)
import pickle as pkl
import sklearn
import numpy as np

@app.route('/covid/', methods=['GET'])
def respond():
    # Retrieve the arguements from url parameter
    receivedDict = request.args
    # For debugging
    # print(f"receivedDict {receivedDict}")

    rawList = []

    for item in receivedDict:
        rawList.append(receivedDict[item])

    clf = pkl.load(open('covidmodel', 'rb'))
    inputVector = np.array(rawList)
    inputVector = inputVector.reshape(1,-1)

    y = clf.predict(inputVector)

    ans = ""

    if(y[0] == 0):
        ans = "0"
    else:
        ans = "1"

    
    # Return the response in json format
    return ans

@app.route('/heart/', methods=['GET'])
def respond():
    # Retrieve the arguements from url parameter
    receivedDict = request.args
    # For debugging
    # print(f"receivedDict {receivedDict}")

    rawList = []

    for item in receivedDict:
        rawList.append(receivedDict[item])

    clf = pkl.load(open('heartmodel', 'rb'))
    inputVector = np.array(rawList)
    inputVector = inputVector.reshape(1,-1)

    y = clf.predict(inputVector)

    ans = ""

    if(y[0] == 0):
        ans = "0"
    else:
        ans = "1"

    
    # Return the response in json format
    return ans



# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
