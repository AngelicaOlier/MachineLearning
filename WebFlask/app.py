#pip install -r requirements.txt
#flask --app app run --debug
import re
from datetime import datetime
from flask import Flask, render_template, request
import RL_SalarioExperiencia
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

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