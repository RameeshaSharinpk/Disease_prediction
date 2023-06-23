from flask import Flask, render_template, url_for, request, redirect, session
from flask_mysqldb import MySQL
from pymongo import MongoClient
from flask_restful import Resource, Api
from datetime import date, datetime

import pickle
import bcrypt
import re
import numpy as np
import pandas as pd


app = Flask(__name__)

client = MongoClient("localhost", 27017)

db = client.DiseasePrediction
loaded_model = pickle.load(open("finalized_model.sav", "rb"))
final_rf_model = loaded_model["final_model"]
symptoms = loaded_model["symptoms"]
data_dict = loaded_model["data_dict"]

# session['mail'] = ""

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        message = ""
        users = db.user
        login_user = users.find_one({"email": request.form["email"]})

        if login_user:
            if bcrypt.hashpw(request.form['password'].encode('utf-8'), login_user['password']) == login_user['password']:
                session['email'] = request.form['email']
                log_name = login_user['name'].capitalize()
                session['mail'] = login_user['email']
                return render_template('predict.html', log_name = log_name)
        message = "Invalid email/password combination"
        return render_template("login.html", message = message)
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    message = ""
    if request.method == "POST":
        users = db.user
        existing_user = users.find_one({"email": request.form["email"]})

        if existing_user is None:
            hashpass = bcrypt.hashpw(
                request.form["password"].encode("utf-8"), bcrypt.gensalt()
            )
            users.insert_one(
                {
                    "name": request.form["name"],
                    "email": request.form["email"],
                    "password": hashpass,
                }
            )
            session["email"] = request.form["email"]
            log_name = request.form['name'].capitalize()
            session['mail'] = request.form['email']
            return render_template('/login', log_name = log_name)
        message = "The user already exists!"
        return render_template("signup.html", message=message)
    return render_template("signup.html")




@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    # %matplotlib inline
    session['input_symptoms'] = ",".join(request.form["symptoms"][:-2].split(", "))
    

    def predictDisease(symptoms):
        # symptoms = [symptom1,symptom2,symptom3,symptom4,symptom5,symptom6]
        symptoms = symptoms.split(",")

        # creating input data for the models
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            symptom_transformed = "_".join([i.lower() for i in symptom.split(" ")])
            index = data_dict["symptom_index"][symptom_transformed]
            input_data[index] = 1

        input_data = np.array(input_data).reshape(1, -1)
        # generating individual outputs
        rf_prediction = data_dict["predictions_classes"][
            final_rf_model.predict(input_data)[0]
        ]
        return rf_prediction
    # input_disease = [symptom1,symptom2,symptom3,symptom4,symptom5,symptom6]
    # print(input_symptoms)
    #
    # Testing the function
    session['result'] = predictDisease(session['input_symptoms'])
    # mail = session.get('mail', None)
    print(session['mail'])
    users = db.user
    found_user = users.find_one({'email': session['mail']})
    # sympt = []
    # diseas = []

    if found_user:
        current_date = date.today().strftime('%m/%d/%Y')
        symptoms = found_user.get('symptoms', [])
        symptoms.append(session['input_symptoms'])
        diseases = found_user.get('disease', [])
        diseases.append(session['result'])
        dates = found_user.get('dates',[])
        dates.append(current_date)
        users.update_one(
            {'email': session['email']},
            {'$set': {'symptoms': symptoms, 'disease': diseases, 'dates':dates}}
        )

    session['result'] =session['result']
    
    return render_template("predict.html", prediction=session['result'])

@app.route("/history")
def history():
    users = db.user
    found_user = users.find_one({'email': session['email']})

    if found_user:
        symptoms = found_user.get('symptoms', [])
        diseases = found_user.get('disease', [])
        dates = found_user.get('dates',[])
        

    return render_template('history.html', symptoms=symptoms, diseases=diseases, date=dates)

if __name__ == "__main__":
    app.secret_key = "mysecret"
    app.run(host="127.0.0.1", port=8000, debug=True)
