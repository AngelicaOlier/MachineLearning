#py -m install flask 
#py -m flask run 
from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar dataset
datos_dataset= [
    ["kilometraje", "temperatura_motor", "mantenimiento", "componente", "fallo"],
    [11823, 98.39416201136112, 0, 'motor', 1],
    [33884, 99.56571411793459, 1, 'suspension', 0],
    [188019, 82.5450347342996, 0, 'suspension', 1],
    [171714, 81.05086031218141, 1, 'transmision', 1],
    [38271, 117.37070664890118, 1, 'transmision', 1],
    [29839, 117.2184147406769, 0, 'transmision', 1],
    [34574, 110.72718617718368, 0, 'frenos', 1],
    [28199, 60.91023641351857, 1, 'transmision', 1],
    [184533, 77.59844432340562, 1, 'motor', 0],
    [73862, 111.12681284743452, 1, 'suspension', 0],
    [176513, 117.15764347480895, 0, 'transmision', 1],
    [78793, 111.28022784281336, 0, 'frenos', 0],
    [146701, 67.07755411475429, 0, 'motor', 0],
    [150466, 107.55203841019198, 1, 'transmision', 0],
    [64428, 78.17768029827073, 0, 'transmision', 1],
    [92126, 106.05787757582249, 0, 'motor', 0],
    [126288, 90.79034633211774, 0, 'suspension', 0],
    [58444, 78.37560628595355, 0, 'motor', 1],
    [56499, 73.52747715817428, 1, 'transmision', 1],
    [138013, 82.35637918709509, 0, 'motor', 0],
    [23967, 87.29576821395618, 0, 'suspension', 1],
    [160537, 84.25100032837797, 0, 'frenos', 0],
    [68764, 108.48025488766825, 0, 'transmision', 1],
    [152542, 94.00131953027898, 0, 'motor', 1],
    [178393, 118.561617843512, 0, 'frenos', 1],
    [123722, 109.0902583593711, 0, 'transmision', 1],
    [22711, 105.43835427972576, 1, 'frenos', 0],
    [162154, 75.75221980362143, 1, 'transmision', 0],
    [44622, 91.05985751807228, 1, 'suspension', 0],
    [127428, 70.23830703052711, 1, 'frenos', 0],
    [194345, 101.06549633502996, 1, 'motor', 1],
    [168870, 98.52279310666457, 1, 'frenos', 0],
    [195241, 86.04746112082411, 1, 'frenos', 1],
    [103545, 86.61764243167316, 1, 'motor', 0],
    [98162, 85.91010308577144, 0, 'motor', 0],
    [123228, 82.4238648488675, 1, 'transmision', 1],
    [148118, 110.08503063376787, 1, 'suspension', 1],
    [97165, 80.81046731778933, 1, 'frenos', 0],
    [31973, 115.72321357164772, 0, 'suspension', 0],
    [156136, 90.9446472181683, 1, 'transmision', 0],
    [111859, 96.42933045829986, 0, 'suspension', 1],
    [118378, 64.62812300796207, 0, 'suspension', 1],
    [199060, 95.60266750831703, 1, 'transmision', 0],
    [67870, 90.6419372473861, 1, 'transmision', 0],
    [141389, 85.58560936604151, 0, 'transmision', 1],
    [49388, 109.5686076833907, 1, 'frenos', 0],
    [191197, 62.16388137063889, 0, 'motor', 1],
    [143858, 104.03748388630758, 0, 'frenos', 0],
    [104620, 115.47753485903695, 0, 'frenos', 1],
    [197556, 90.5126068290316, 0, 'suspension', 0],
    [24344, 89.78143991991018, 1, 'frenos', 0],
    [33391, 114.27891193940997, 1, 'motor', 1],
    [145967, 94.01519747344145, 1, 'suspension', 1],
    [140251, 75.80634997918362, 1, 'motor', 1],
    [164628, 119.63471219245704, 1, 'frenos', 1],
    [22773, 64.09918011844975, 0, 'motor', 0],
    [136905, 105.256006953465, 0, 'transmision', 0],
    [56142, 75.22199763151454, 1, 'suspension', 1],
    [48487, 98.00160115693006, 0, 'motor', 1],
    [143501, 85.12521904510196, 0, 'frenos', 1],
    [148692, 114.20245514298716, 0, 'frenos', 1],
    [175157, 107.79576286777322, 0, 'transmision', 1],
    [195901, 83.50790418720753, 0, 'frenos', 0],
    [164815, 101.14809543110059, 1, 'motor', 1],
    [14406, 105.11385035299277, 0, 'frenos', 1],
    [105672, 75.77495106880826, 0, 'transmision', 0],
    [58921, 67.84273484115342, 0, 'suspension', 1],
    [134325, 95.921176149522, 0, 'transmision', 0],
    [101506, 79.53670983303212, 0, 'frenos', 0],
    [159008, 115.28623361275083, 1, 'transmision', 1],
    [116140, 116.8253539558564, 0, 'frenos', 1],
    [43285, 115.51458047945943, 0, 'suspension', 1],
    [43591, 118.95514673601403, 0, 'suspension', 1],
    [100069, 112.21194036551145, 1, 'motor', 0],
    [147409, 94.31869588735695, 1, 'motor', 1],
    [173466, 114.96543084048257, 1, 'motor', 0],
    [74246, 85.4631092387032, 0, 'frenos', 0],
    [93004, 95.16869595611097, 1, 'motor', 1],
    [189460, 105.11779542782457, 1, 'suspension', 1],
    [48188, 89.6220682431793, 1, 'suspension', 0],
    [148355, 80.99586851132877, 0, 'suspension', 1],
    [86313, 85.99779903579955, 0, 'motor', 1],
    [114706, 109.45556979208335, 1, 'motor', 0],
    [164837, 97.7661269038812, 0, 'frenos', 1],
    [173978, 100.1727875407718, 1, 'suspension', 0],
    [113299, 111.6262648726873, 0, 'motor', 1],
    [156772, 106.86980365783172, 0, 'suspension', 1],
    [105857, 107.53328415889123, 1, 'motor', 0],
    [198999, 78.20458464179894, 0, 'frenos', 1],
    [10470, 116.2675799486461, 0, 'motor', 0],
    [15321, 107.0612040426354, 1, 'motor', 0],
    [156293, 78.7974768167695, 1, 'suspension', 1],
    [122560, 113.90561458805788, 0, 'motor', 1],
    [113502, 99.3545359347288, 0, 'frenos', 1],
    [134663, 95.58677156963744, 1, 'suspension', 0],
    [174925, 106.68902892667196, 0, 'motor', 1],
    [128919, 90.04913626180223, 0, 'suspension', 1],
    [158993, 106.8985712361766, 0, 'frenos', 1],
    [161131, 98.6586544889729, 1, 'motor', 0],
    [157931, 107.0884161198964, 0, 'motor', 1],
    [150152, 82.68806584807496, 0, 'suspension', 1],
    [118622, 109.5698629057139, 0, 'suspension', 1],
    [174682, 95.6269624272467, 1, 'motor', 0],
    [167083, 75.7987127997276, 1, 'motor', 0],
    [128270, 105.54722254677902, 0, 'frenos', 1],
    [153606, 107.54948642193204, 1, 'motor', 0],
    [156971, 84.2039494892188, 0, 'suspension', 1],
    [167788, 107.04286555605398, 0, 'motor', 1]
    ]

# Convertir el dataset en arrays
data_array = np.array(datos_dataset[1:])  # Excluir la fila de títulos

# Extraer características y variable objetivo
X = data_array[:, :2].astype(float)  # Suponiendo que las características son las dos primeras columnas
y = data_array[:, -1].astype(int)  # Suponiendo que la variable objetivo es la última columna

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo
model = LogisticRegression()
model.fit(X_train, y_train)

def predict_failure(kilometraje, temperatura_motor, mantenimiento, componente):
    """Predice la probabilidad de falla según los factores dados."""
    input_data = np.array([[kilometraje, temperatura_motor]])
    input_data = scaler.transform(input_data)
    
    # Calculamos la probabilidad y la convertimos a porcentaje
    probability = model.predict_proba(input_data)[0][1] * 100
    
    # Determinamos si hay fallo (1) o no (0)
    prediction = "Fallo probable" if probability > 50 else "Sin riesgo significativo"
    
    return prediction, round(probability, 2)

def generate_graph():
    """Genera un gráfico de los datos y la frontera de decisión."""
    plt.figure(figsize=(8, 5))
    
    # Graficar los datos
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    
    # Configuración del gráfico
    plt.xlabel("Kilometraje")
    plt.ylabel("Temperatura Motor")
    plt.title("Regresión Logística: Predicción de Fallas")
    plt.colorbar(label="Probabilidad de Falla")
    
    # Guardar la imagen como base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()
    
    return image_base64