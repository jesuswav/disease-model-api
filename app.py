from flask import Flask, request, jsonify
from keras.models import load_model
from utils import preprocess_image
import numpy as np
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Crear app Flask
app = Flask(__name__)

CORS(app)  # 🔓 Habilita CORS globalmente

# Cargar el modelo
MODEL_PATH = "modelo.keras"
model = load_model(MODEL_PATH)

# Diccionario de clases
class_dict = {
    'Maize fall armyworm': 0,
    'Maize grasshoper': 1,
    'Maize healthy': 2,
    'Maize leaf beetle': 3,
    'Maize leaf blight': 4,
    'Maize leaf spot': 5,
    'Maize streak virus': 6,
    'Tomato healthy': 7,
    'Tomato leaf blight': 8,
    'Tomato leaf curl': 9,
    'Tomato septoria leaf spot': 10,
    'Tomato verticulium wilt': 11
}
# Invertir el diccionario para mapear índices a nombres
idx_to_class = {v: k for k, v in class_dict.items()}

# Ruta para recibir imágenes y hacer predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    file.save(file_path)

    try:
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        class_name = idx_to_class.get(predicted_class, "Clase desconocida")

        return jsonify({
            'predicted_class_index': predicted_class,
            'predicted_class_name': class_name,
            'confidence': round(confidence, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(file_path)

# Ruta raíz
@app.route('/', methods=['GET'])
def index():
    return "API para predicción de imágenes con modelo Keras"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)