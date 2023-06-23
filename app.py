from flask import Flask, render_template, url_for, request, redirect, session
from flask_mysqldb import MySQL
from pymongo import MongoClient
from flask_restful import Resource, Api

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
                return render_template('home.html', log_name = log_name)
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
                    "history":[]
                }
            )
            session["email"] = request.form["email"]
            log_name = request.form['name'].capitalize()
            mail = request.form['email']
            return render_template('/home', log_name = log_name)
        message = "The user already exists!"
        return render_template("signup.html", message=message)
    return render_template("signup.html")




@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    # %matplotlib inline
    input_symptoms = ",".join(request.form["symptoms"][:-2].split(", "))
    # print(request.form["symptoms"])
    # sympt = []
    # symptom1 = request.form["symptom1"]
    # symptom2 = request.form["symptom2"]
    # symptom3 = request.form["symptom3"]
    # symptom4 = request.form["symptom4"]
    # symptom5 = request.form["symptom5"]
    # symptom6 = request.form["symptom6"]
    # input_symptoms = symptom1+","+symptom2+","+symptom3+","+symptom4+","+symptom5+","+symptom6

    # Creating a symptom index dictionary to encode the
    # input symptoms into numerical form

    # Defining the Function
    # Input: string containing symptoms separated by commas
    # Output: Generated predictions by models

    def predictDisease(symptoms):
        
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
    result = predictDisease(input_symptoms)
    # mail = session.get('mail', None)
    print(session['mail'])
    users = db.user
    
    users.find_one_and_update({"email": session['mail']},{'$set':{
        "history": {
            "symptoms": input_symptoms,
            "disease": result
        }}
    },upsert = True)
    return render_template("predict.html", prediction=result)

@app.route("/home")
def home():
    return render_template('home.html')


if __name__ == "__main__":
    app.secret_key = "mysecret"
    app.run(host="127.0.0.1", port=8000, debug=True)
