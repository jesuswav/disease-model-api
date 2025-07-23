import numpy as np
from PIL import Image

def preprocess_image(image_path):
    """
    Carga una imagen y la convierte a tensor compatible con el modelo
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))  # Ajusta según tu modelo
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # [1, 224, 224, 3]
    return img_array