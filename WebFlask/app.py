#pip install -r requirements.txt
#flask --app app run --debug
import re
from datetime import datetime
from flask import Flask, render_template, request
import RL_SalarioExperiencia
import RLg_SectorAutomotriz
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/MlCasoDeUsoSupervisado')
def MlCaso():
    return render_template('MlCasoDeUsoSupervisado.html')

@app.route("/linearRegressionExpSalario", methods=["GET", "POST"])
def linear_regression_Exp_Salario():
    predicted_result = None
    graph_image = RL_SalarioExperiencia.generate_graph()

    if request.method == "POST":
        try:
            experiencia = float(request.form.get("Experiencia"))
            predicted_result = RL_SalarioExperiencia.model.predict([[experiencia]])[0]
        except ValueError:
            predicted_result = "Entrada no válida"

    return render_template("linearRegressionExpSalario.html",
                           result=predicted_result,
                           graph=graph_image)
    
@app.route('/mapa')
def map():
    return render_template('mapa.html')


@app.route("/logisticRegressionFallo", methods=["GET", "POST"])
def logistic_regression_fallo():
    predicted_result = None
    probability = None
    
    if request.method == "POST":
        try:
            kilometraje = float(request.form.get("kilometraje"))
            temperatura_motor = float(request.form.get("temperatura_motor"))
            mantenimiento = int(request.form.get("mantenimiento"))
            componente = request.form.get("componente")
            
            predicted_result, probability = RLg_SectorAutomotriz.predict_failure(kilometraje, temperatura_motor, mantenimiento, componente)
        except ValueError:
            predicted_result = "Entrada no válida"

    return render_template("logisticRegressionFallo.html",
                           result=predicted_result,
                           probability=probability)

if __name__ == '__main__':
    app.run(debug=True)