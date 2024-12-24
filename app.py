from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import keras

app = Flask(__name__)
CORS(app)

def load_model_without_batch_shape(model_path):
    model = keras.models.load_model(model_path)
    for layer in model.layers:
        if hasattr(layer, 'batch_input_shape'):
            layer.batch_input_shape = None
    return model

model = load_model_without_batch_shape('saved_model.h5')

def preprocess_image(image):
    image = image.resize((100, 100))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file)
        img = preprocess_image(img)
        class_mapping = {0: "Parasitized", 1: "Uninfected"}
        predictions = model.predict(img)    
        predicted_labels = (predictions > 0.5).astype("int32")
        predicted_class_names = [class_mapping[label] for label in predicted_labels.flatten()]
        return jsonify({'prediction': predicted_class_names[0]})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
