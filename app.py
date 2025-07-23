from flask import Flask, request, jsonify
from keras.models import load_model
from utils import preprocess_image
import numpy as np
import os
from werkzeug.utils import secure_filename

# Crear app Flask
app = Flask(__name__)

# Cargar el modelo
MODEL_PATH = "modelo.keras"
model = load_model(MODEL_PATH)

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

        return jsonify({
            'predicted_class': predicted_class,
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