import matplotlib.pylab as plt
from API import transfer_style
import os

if __name__ == "__main__":

    # Path of the pre-trained TF model
    model_path = r"C:\Users\Admin\Neural-Style-Transfer\model"

    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")

    # NOTE: Works for common image formats such as '.jpg', '.jpeg', '.png', '.bmp', '.tiff'
    content_image_path = r"C:\Users\Admin\Neural-Style-Transfer\Imgs\content3.jpg"
    style_image_path = r"C:\Users\Admin\Neural-Style-Transfer\Imgs\art2.png"

    # Check if image paths exist
    if not os.path.exists(content_image_path):
        raise FileNotFoundError(f"Content image path {content_image_path} does not exist")
    if not os.path.exists(style_image_path):
        raise FileNotFoundError(f"Style image path {style_image_path} does not exist")

    # Get user input for style percentage
    try:
        style_percentage = float(input("Enter style percentage (0.0 to 1.0): "))
    except ValueError:
        raise ValueError("Invalid input. Style percentage must be a float between 0.0 and 1.0")

    # Validate user input
    if style_percentage < 0 or style_percentage > 1:
        raise ValueError("Style percentage must be between 0.0 and 1.0")

    img = transfer_style(content_image_path, style_image_path, model_path, style_percentage)

    # Saving the generated image
    plt.imsave('stylized_image.jpeg', img)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
