#pip install -r requirements.txt
#flask --app app run --debug
import re
from datetime import datetime
from flask import Flask, render_template, request
import RL_SalarioExperiencia

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

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

#SEMANA 7
@app.route("/RL")
def rl():
    return render_template("S7/RegresionLogistica.html")
@app.route("/KNN")
def knn():
    return render_template("S7/KNN.html")
@app.route("/Arbol")
def arbol():
    return render_template("S7/ArbolesDecision.html")
@app.route("/RF")
def rf():
    return render_template("S7/RandomForest.html")
@app.route("/SVM")
def svm():
    return render_template("S7/SVM.html")
@app.route("/GB")
def gb():
    return render_template("S7/GradientBoosting.html")
@app.route("/NB")
def nb():
    return render_template("S7/NaiveBayes.html")