import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np

original_images_dir = os.path.abspath(r"D:\Mestrado\ppgca_ia_consumer\app\main\tests\data\cropped_images")
original_images = sorted([f for f in os.listdir(original_images_dir) if os.path.isfile(os.path.join(original_images_dir, f))])

def clahe(image: Image.Image, clip_limit=16, tile_grid_size=(8, 8)) -> Image.Image:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the input image.
    
    :param image: Input image in PIL format
    :param clip_limit: Threshold for contrast limiting (default 16)
    :param tile_grid_size: Size of grid for histogram equalization (default (8, 8))
    :return: Image with CLAHE applied in PIL format
    """
    image_cv = np.array(image)

    if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=10)
    cl1 = clahe.apply(image_cv)
    clahe_image_rgb = cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB)
    clahe_image_pil = Image.fromarray(clahe_image_rgb)

    return clahe_image_pil

for image_file in original_images:
    image_path = os.path.join(original_images_dir, image_file)
    
    image = Image.open(image_path)
    
    clahe_image = clahe(image, clip_limit=16, tile_grid_size=(8, 8))
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    ax[1].imshow(clahe_image)
    ax[1].set_title("CLAHE Image")
    ax[1].axis("off")
    
    plt.show()
