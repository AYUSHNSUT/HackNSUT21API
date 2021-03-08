# app.py
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium import webdriver
from bs4 import BeautifulSoup
import re
import requests
import numpy as np
import sklearn
import pickle as pkl
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
cors = CORS(app)


@app.route('/medicine', methods=['GET'])
def isabel():
    if request.method == 'GET':
        i = request.args.get("medicine_name")
        print(i)
        # condition if any of the params is missing
        if i == None:
            return jsonify({'error': 'name of medicine is missing!!'}), 400

        SAUCE_USERNAME = 'kapishgarg0214'
        SAUCE_ACCESS_KEY = '3cf56841-a783-4661-8f91-27bbe9aa5c35'

        print("starting the session")
        driver = webdriver.Remote(
            desired_capabilities=webdriver.DesiredCapabilities.FIREFOX,
            command_executor='http://%s:%s@ondemand.saucelabs.com:80/wd/hub' %
            (SAUCE_USERNAME, SAUCE_ACCESS_KEY)
        )

        driver.get("https://www.1mg.com/search/all?name="+str(i))

        data = driver.page_source

        driver.quit()

        soup = BeautifulSoup(data, 'html5lib')
        prices = []
        names = []
        main_container = soup.find_all(
            "div", {"class": "row style__grid-container___3OfcL"})
        if(len(main_container) == 0):
            return jsonify([], []), 200
        if(len(main_container) >= 2):
            all_drugs = main_container[1].find_all(
                "div", {"class": "col-xs-12 col-md-9 col-sm-9 style__container___cTDz0"})
        else:
            all_drugs = main_container[0].find_all(
                "div", {"class": "col-xs-12 col-md-9 col-sm-9 style__container___cTDz0"})
        for drug in all_drugs:
            drug_name = drug.find(
                "div", {"class": "style__product-description___1vPQe"}).text
            price = drug.find(
                "div", {"class": "style__price-tag___B2csA"}).text
            price = re.sub('[^0-9.]', "", price)
            names.append(drug_name)
            prices.append(price)
        ans = [names, prices]
        return jsonify(ans), 200


@app.route('/covid/', methods=['GET'])
def respond():
    # Retrieve the arguements from url parameter
    receivedDict = request.args
    # For debugging
    # print(f"receivedDict {receivedDict}")

    rawList = []

    for item in receivedDict:
        rawList.append(float(receivedDict[item]))

    clf = pkl.load(open('covidmodel', 'rb'))
    inputVector = np.array(rawList)
    inputVector = inputVector.reshape(1, -1)

    y = clf.predict(inputVector)

    ans = ""

    if(y[0] == 0):
        ans = "0"
    else:
        ans = "1"

    # Return the response in json format
    return ans


@app.route('/heart/', methods=['GET'])
def respond_heart():
    # Retrieve the arguements from url parameter
    receivedDict = request.args
    # For debugging
    # print(f"receivedDict {receivedDict}")

    rawList = []

    for item in receivedDict:
        rawList.append(float(receivedDict[item]))

    clf = pkl.load(open('heartmodel', 'rb'))
    inputVector = np.array(rawList)
    inputVector = inputVector.reshape(1, -1)

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
