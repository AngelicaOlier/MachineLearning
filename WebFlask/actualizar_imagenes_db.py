# usado para cargar las imágenes en la base de datos 
import psycopg2
import os

# --- CONFIGURACIÓN ---
db_params = {
    'dbname': 'DB_ML_nueva',  # nombre de la base de datos
    'user': 'postgres',           # usuario de PostgreSQL
    'password': '123456',         # contraseña
    'host': 'localhost',          # la dirección del host de tu DB (Render te dará una)
    'port': '5432'                # Puerto estándar de PostgreSQL
}

imagenes_a_actualizar = {
    1: 'static/images/S7/NB.png',         
    2: 'static/images/S7/GradientBoosting.png',   
    3: 'static/images/S7/SVM.png',                
    4: 'static/images/S7/randomforest.png',     
    5: 'static/images/S7/ARBOLDESICIONES.png',  
    6: 'static/images/S7/KNN.png',                
    7: 'static/images/S7/RL.png',         
}


# --- LÓGICA DE ACTUALIZACIÓN by IA---
conn = None
cur = None
actualizaciones_exitosas = 0
errores = 0

try:
    print("Conectando a la base de datos PostgreSQL...")
    # Descomenta la línea de abajo si usas DATABASE_URL
    # conn = psycopg2.connect(DATABASE_URL)
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    print("Conexión exitosa.")

    for id_modelo, ruta_archivo in imagenes_a_actualizar.items():
        print(f"\nProcesando modelo ID: {id_modelo}")
        print(f"  Ruta de imagen: {ruta_archivo}")

        if not os.path.exists(ruta_archivo):
            print(f"  ERROR: Archivo de imagen no encontrado en '{ruta_archivo}'. Saltando.")
            errores += 1
            continue

        try:
            # Leer el archivo de imagen en modo binario ('rb')
            with open(ruta_archivo, "rb") as f:
                imagen_binaria = f.read()
            print(f"  Imagen leída ({len(imagen_binaria)} bytes).")

            # Preparar el comando UPDATE
            sql_update = """
                UPDATE modelo
                SET imagen = %s
                WHERE id_modelo = %s;
            """

            # Ejecutar el UPDATE (psycopg2 maneja bytes -> BYTEA)
            cur.execute(sql_update, (imagen_binaria, id_modelo))
            print(f"  UPDATE ejecutado para id_modelo = {id_modelo}.")
            actualizaciones_exitosas += 1

        except (Exception, psycopg2.Error) as error_individual:
            print(f"  ERROR al actualizar id_modelo {id_modelo}: {error_individual}")
            conn.rollback() # Deshacer cambios para este modelo si falla
            errores += 1
        else:
             # Solo hacer commit si este modelo individual se actualizó bien
             conn.commit()
             print(f"  Commit realizado para id_modelo = {id_modelo}.")


except (Exception, psycopg2.DatabaseError) as error:
    print(f"\nError de base de datos o conexión: {error}")
    if conn:
        conn.rollback() # Deshacer cualquier cambio pendiente si la conexión falla
    errores += 1 # Contar como error general

finally:
    print("\n--- Resumen ---")
    print(f"Actualizaciones exitosas: {actualizaciones_exitosas}")
    print(f"Errores: {errores}")
    if cur:
        cur.close()
    if conn:
        conn.close()
        print("Conexión a la base de datos cerrada.")