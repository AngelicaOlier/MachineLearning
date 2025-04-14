from flask import Flask # se añade Response para las imagenes
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import os

load_dotenv()  # Cargar variables de entorno desde el archivo .env
#------------------------------------------
app = Flask(__name__)
#------------------------------------------
# --- CONFIGURACIÓN DE LA BASE DE DATOS ---
DATABASE_URL = (
    f"{os.getenv('BASE_DE_DATOS')}://{os.getenv('USUARIO_BD')}:"
    f"{os.getenv('CONTRASENA_BD')}@{os.getenv('HOST_BD')}:"
    f"{os.getenv('PUERTO_BD')}/{os.getenv('NOMBRE_BD')}"
)
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

engine = create_engine(DATABASE_URL, echo=True, connect_args={'client_encoding': 'utf8'})
session = Session(engine)

# --- INICIALIZACIÓN DE EXTENSIONES ---
db = SQLAlchemy(app)
migrate = Migrate(app, db)

__all__ = ['app', 'db', 'migrate', 'session']
