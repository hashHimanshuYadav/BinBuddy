from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# Setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = tf.keras.models.load_model('models/waste_classifier.h5')

# Load class names from training folder
train_dir = 'dataset/train'
class_names = sorted(os.listdir(train_dir))

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_class = None
    img_path = None
    if request.method == 'POST':
        # Upload image
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            # Predict
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = model.predict(img_array)
            class_idx = np.argmax(preds)
            pred_class = class_names[class_idx]

    return render_template('index.html', pred_class=pred_class, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)