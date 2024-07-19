import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os

def transfer_style(content_image, style_image, model_path, style_percentage=1.0):
    """
    :param content_image: path of the content image
    :param style_image: path of the style image
    :param model_path: path to the downloaded pre-trained model.
    :param style_percentage: float, how much percentage of the style should be transferred. Range [0, 1].

    The 'model' directory already contains the downloaded pre-trained model, but 
    you can also download the pre-trained model from the below TF HUB link:
    https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2

    :return: An image as 3D numpy array.
    """

    print("Loading images...")
    # Load content and style images using OpenCV
    content_image = cv2.imread(content_image)
    style_image = cv2.imread(style_image)

    # Check if images are loaded properly
    if content_image is None:
        raise FileNotFoundError(f"Content image {content_image} could not be loaded")
    if style_image is None:
        raise FileNotFoundError(f"Style image {style_image} could not be loaded")

    # Convert BGR (OpenCV default) to RGB
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)

    print("Resizing and Normalizing images...")
    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

    # Optionally resize the images. It is recommended that the style image is about
    # 256 pixels (this size was used when training the style transfer network).
    # The content image can be any size.
    style_image = tf.image.resize(style_image, (256, 256))

    print("Loading pre-trained model...")
    # The hub.load() loads any TF Hub model
    hub_module = hub.load(model_path)

    print("Generating stylized image now...wait a minute")
    # Stylize image.
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0].numpy()

    print("Blending the stylized image with the content image...")
    # Blend the stylized image with the original content image
    stylized_image = style_percentage * stylized_image + (1 - style_percentage) * content_image

    # Reshape the stylized image
    stylized_image = np.array(stylized_image)
    stylized_image = stylized_image.reshape(
        stylized_image.shape[1], stylized_image.shape[2], stylized_image.shape[3])

    print("Stylizing completed...")
    return stylized_image
