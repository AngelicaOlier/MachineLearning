import re
from datetime import datetime
from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/MlCasoDeUsoSupervisado')
def MlCaso():
    return render_template('MlCasoDeUsoSupervisado.html')

if __name__ == '__main__':
    app.run(debug=True)