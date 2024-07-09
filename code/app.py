# importing the libraries 
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template

df = pd.read_csv('data/anemia.csv')
print(df.head())

print(df.info())

@app.route('/predict', methods=["POST"])
def predict():

  Gender = float(request.form["Gender"])
  Hemoglobin = float(request.form["Hemoglobin"])
  MCH = float(request.form["MCH"])
  MCHC = float(request.form["MCHC"])
  MCV = float(request.form["MCV"])

  features_values = np.array([[Gender,Hemoglobin,MCH,MCHC,MCV]])

df = pd.DataFrame(features_values, columns=['Gender','Hemoglobin','MCH','MCHC','MCV'])
print(df)

prediction = model.predict(df)
print(prediciton[0])
result = predicition[0]

if predicition[0] == 0:
  result = "You don't have any Anemic Disease"
elif prediction[0] == 1:
  result = "You have anemic disease"

text = "Hence, based on calculation: "
return render_template("predict.html, prediction_text=text + str(result))
