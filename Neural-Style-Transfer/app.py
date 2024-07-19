from flask import Flask, request, send_file, render_template
import os
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Path to the pre-trained model
MODEL_PATH = r"C:\Users\Admin\Neural-Style-Transfer\model"

# Load the model
hub_module = hub.load(MODEL_PATH)

def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image

def transfer_style(content_image_path, style_image_path, style_percentage=1.0):
    content_image = preprocess_image(content_image_path)
    style_image = preprocess_image(style_image_path)

    content_image = np.expand_dims(content_image, axis=0)
    style_image = np.expand_dims(style_image, axis=0)
    style_image = tf.image.resize(style_image, (256, 256))

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0].numpy()

    stylized_image = style_percentage * stylized_image + (1 - style_percentage) * content_image

    stylized_image = stylized_image.squeeze()
    stylized_image = np.clip(stylized_image, 0, 1)

    return stylized_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return "Missing files", 400

    content_image = request.files['content_image']
    style_image = request.files['style_image']
    style_percentage = float(request.form.get('style_percentage', 1.0))

    content_image_path = os.path.join("uploads", content_image.filename)
    style_image_path = os.path.join("uploads", style_image.filename)

    content_image.save(content_image_path)
    style_image.save(style_image_path)

    stylized_image = transfer_style(content_image_path, style_image_path, style_percentage)

    img_io = BytesIO()
    plt.imsave(img_io, stylized_image, format='jpeg')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
