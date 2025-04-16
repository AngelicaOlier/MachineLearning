#pip install -r requirements.txt
#flask --app app run --debug
from flask import render_template, request, Response # se añade Response para las imagenes
from sqlalchemy import text
from conexionDB import app,session,db
from MLClasificacionCultivo import entrenar_evaluar_modelo_cultivo 
import RL_SalarioExperiencia
import RLg_SectorAutomotriz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#------------------------------------------
@app.route("/conexion_mysql")
def conexion_mysql_prueba():
    try:
        session.execute(text("SELECT 1"))
        print ("conectado")
    except Exception as e:
        print (f"Error de conexión: {e}")
    return "Prueba de conexión a MySQL"
#------------------------------------------

# --- DEFINICIÓN DE MODELOS SQLAlchemy ---
class TipoModelo(db.Model):
    __tablename__ = 'tipomodelo'
    id_tipomodelo = db.Column(db.Integer, primary_key=True)
    tipo = db.Column(db.String(100), unique=True, nullable=False)
    modelos = db.relationship('Modelo', backref='tipo', lazy=True)

class Modelo(db.Model):
    __tablename__ = 'modelo'
    id_modelo = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(255), unique=True, nullable=False)
    imagen = db.Column(db.LargeBinary, nullable=True)
    id_tipomodelo = db.Column(db.Integer, db.ForeignKey('tipomodelo.id_tipomodelo'), nullable=False)
    informacion = db.relationship('Informacion', backref='modelo', uselist=False, lazy=True, cascade="all, delete-orphan")

class Informacion(db.Model):
    __tablename__ = 'informacion'
    id_informacion = db.Column(db.Integer, primary_key=True)
    descripcion = db.Column(db.Text, nullable=False)
    id_modelo = db.Column(db.Integer, db.ForeignKey('modelo.id_modelo'), unique=True, nullable=False)
    fuentes = db.relationship('FuentesInformacion', backref='informacion', lazy=True, cascade="all, delete-orphan")

class FuentesInformacion(db.Model):
    __tablename__ = 'fuentes_informacion'
    id_fuente = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.Text, nullable=False)
    id_informacion = db.Column(db.Integer, db.ForeignKey('informacion.id_informacion'), nullable=False)
    descripcion_link = db.Column(db.String(255), nullable=True)
#------------------------------------------

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

#SEMANA 7
# --- FUNCIÓN AUXILIAR ---
def obtener_datos_modelo(id_modelo_buscar):
    """Consulta la BD y devuelve el objeto Modelo con sus relaciones."""
    try:
        modelo = Modelo.query.options(
            db.joinedload(Modelo.tipo),
            db.joinedload(Modelo.informacion).joinedload(Informacion.fuentes)
        ).get_or_404(id_modelo_buscar) # get_or_404 es más directo que filter_by + first_or_404 para PK
        return modelo
    except Exception as e:
        print(f"Error al consultar el modelo ID {id_modelo_buscar}: {e}")
        return None 
#------------------------------------------
# --- RUTAS PARA CADA TIPO DE MODELO ---    
@app.route("/RL") # Ruta para Regresión Logística
def rl():
    modelo_datos = obtener_datos_modelo(id_modelo_buscar=7) # <- ID de Regresión Logística
    if modelo_datos is None:
         return "Error al cargar la información del modelo.", 500
    return render_template("S7/detallesModelo.html", modelo=modelo_datos)

@app.route("/KNN") # Ruta para KNN
def knn():
    modelo_datos = obtener_datos_modelo(id_modelo_buscar=6) # <- ID de KNN 
    if modelo_datos is None:
         return "Error al cargar la información del modelo.", 500
    return render_template("S7/detallesModelo.html", modelo=modelo_datos)

@app.route("/Arbol") # Ruta para Árboles de Decisión
def arbol():
    modelo_datos = obtener_datos_modelo(id_modelo_buscar=5) # <- ID de Árboles 
    if modelo_datos is None:
         return "Error al cargar la información del modelo.", 500
    return render_template("S7/detallesModelo.html", modelo=modelo_datos)

@app.route("/RF") # Ruta para Random Forest
def rf():
    modelo_datos = obtener_datos_modelo(id_modelo_buscar=4) # <- ID de Random Forest 
    if modelo_datos is None:
         return "Error al cargar la información del modelo.", 500
    return render_template("S7/detallesModelo.html", modelo=modelo_datos)

@app.route("/SVM") # Ruta para SVM
def svm():
    modelo_datos = obtener_datos_modelo(id_modelo_buscar=3) # <- ID de SVM
    if modelo_datos is None:
         return "Error al cargar la información del modelo.", 500
    return render_template("S7/detallesModelo.html", modelo=modelo_datos)

@app.route("/GB") # Ruta para Gradient Boosting
def gb():
    modelo_datos = obtener_datos_modelo(id_modelo_buscar=2) # <- ID de GB 
    if modelo_datos is None:
         return "Error al cargar la información del modelo.", 500
    return render_template("S7/detallesModelo.html", modelo=modelo_datos)

@app.route("/NB") # Ruta para Naive Bayes
def nb():
    modelo_datos = obtener_datos_modelo(id_modelo_buscar=1) # <- ID de Naive Bayes 
    if modelo_datos is None:
         return "Error al cargar la información del modelo.", 500
    return render_template("S7/detallesModelo.html", modelo=modelo_datos)


# --- RUTA PARA IMÁGENES ---
@app.route('/modelo/<int:id_modelo>/imagen')
def servir_imagen(id_modelo):
    modelo = Modelo.query.get_or_404(id_modelo)
    if modelo.imagen:
        mimetype = 'application/octet-stream' # O intenta ser más específico
        # Ejemplo simple para tipos comunes (requiere instalar 'python-magic' o similar para detección real)
        # if b'\x89PNG' in modelo.imagen[:8]: mimetype = 'image/png'
        # elif b'\xFF\xD8\xFF' in modelo.imagen[:3]: mimetype = 'image/jpeg'
        # elif b'GIF8' in modelo.imagen[:4]: mimetype = 'image/gif'
        return Response(modelo.imagen, mimetype=mimetype)
    else:
         return "No hay imagen disponible", 404


#SEMANA 8 -- ENTRENAR Y EVALUAR EL MODELO DE CULTIVO -- 
@app.route("/MLClasificacionCultivo", methods=["GET", "POST"])
def MLClasificacionCultivo():
    results_data = None
    error_message = None
    if request.method == "POST":
        # Verificar si se incluyó el archivo en la solicitud
        if 'file' not in request.files:
            error_message = "No se seleccionó ningún archivo."
            # No es necesario renderizar aquí si el 'required' del input funciona,
            # pero es una buena validación del lado del servidor.
            return render_template("MLClasificacionCultivo.html", error=error_message)

        file = request.files["file"]

        # Verificar si el nombre del archivo está vacío (el usuario no seleccionó nada)
        if file.filename == '':
            error_message = "No se seleccionó ningún archivo."
            return render_template("MLClasificacionCultivo.html", error=error_message)

        # Verificar si el archivo existe y tiene la extensión correcta
        if file and (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            try:
                # Llamar a la función que ahora entrena, evalúa y guarda
                results_data = entrenar_evaluar_modelo_cultivo(file)
            except Exception as e:
                # Capturar errores de la función de procesamiento
                error_message = str(e)
        else:
            error_message = "Formato de archivo no válido. Por favor, suba un archivo .xlsx o .xls."

    # Renderizar la plantilla, pasando los resultados o el error
    return render_template("MLClasificacionCultivo.html",
                           results_data=results_data,
                           error=error_message)


# siempre al final del archivo
if __name__ == '__main__':
    # Asegúrate de que el directorio 'saved_models_es' exista antes de iniciar la app
    # (aunque entrenar_evaluar_modelo_cultivo también lo crea)
    model_dir_es = "saved_models_es" # Usa la misma variable que en MLClasificacionCultivo.py
    if not os.path.exists(model_dir_es):
        os.makedirs(model_dir_es)
    app.run(debug=True) # debug=True es útil durante el desarrollo