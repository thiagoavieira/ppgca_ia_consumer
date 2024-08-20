import os
import numpy as np
from PIL import Image
from skimage import img_as_ubyte
from skimage.filters import threshold_sauvola, threshold_otsu, threshold_yen
from skimage.filters import try_all_threshold
from skimage.exposure import histogram

import matplotlib.pyplot as plt

def gray_scale(input_data: dict, image) -> tuple:
    """
    Convert the image to grayscale.
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("L"))
    else:
        if len(image.shape) > 2:
            image_np = np.mean(image, axis=2).astype(np.uint8)
        else:
            image_np = image
    
    return input_data, image_np

def sauvola_threshold(input_data: dict, image: Image.Image) -> tuple:
    """
    Applies the Sauvola threshold in the image.
    """
    input_data, image_np = gray_scale(input_data, image)
    
    thresh_sauvola = threshold_sauvola(image_np, window_size=25, k=0.15)
    binary_sauvola = image_np > thresh_sauvola
    
    binary_sauvola_img = Image.fromarray(img_as_ubyte(binary_sauvola))

    return input_data, binary_sauvola_img

def otsu_threshold(input_data: dict, image: Image.Image) -> tuple:
    """
    Applies the Otsu threshold in the image.
    """
    input_data, image_np = gray_scale(input_data, image)
    
    thresh_otsu = threshold_otsu(image_np, nbins=256)
    binary_otsu = image_np > thresh_otsu
    
    binary_otsu_img = Image.fromarray(img_as_ubyte(binary_otsu))
    
    return input_data, binary_otsu_img

def yen_threshold(input_data: dict, image: Image.Image) -> tuple:
    """
    Applies the Otsu threshold in the image.
    """
    input_data, image_np = gray_scale(input_data, image)
    
    thresh_yen = threshold_yen(image_np)
    binary_yen = image_np > thresh_yen
    
    binary_yen_img = Image.fromarray(img_as_ubyte(binary_yen))
    
    return input_data, binary_yen_img

def binarization(input_data: dict, image: Image.Image, threshold=126) -> tuple:
    """
    Realiza a binarização da imagem.
    """
    input_data, image_np = gray_scale(input_data, image)
    
    binary_np = np.where(image_np > threshold, 255, 0).astype(np.uint8)
    
    binary_image = Image.fromarray(binary_np)
    
    return input_data, binary_image

input_data = {}

current_dir = os.path.dirname(os.path.abspath(__file__))
cropped_images_dir = os.path.join(current_dir, '..', 'data', 'cropped_images')
image_mst_path = os.path.join(cropped_images_dir, "97-F-70_block_0_copy.jpg")
image_org_path = os.path.join(cropped_images_dir, "97-F-70_block_0.jpg")

image_mst = Image.open(image_mst_path)
image_org = Image.open(image_org_path)

_, binary_sauvola_img_mst = sauvola_threshold(input_data, image_mst)
_, binary_otsu_img_mst = otsu_threshold(input_data, image_mst)
_, binary_bin_img_mst = binarization(input_data, image_mst)
# _, binary_yen_img_mst = yen_threshold(input_data, image_mst)

_, binary_sauvola_img_org = sauvola_threshold(input_data, image_org)
_, binary_otsu_img_org = otsu_threshold(input_data, image_org)
_, binary_bin_img_org = binarization(input_data, image_org)
# _, binary_yen_img_org = yen_threshold(input_data, image_org)

fig, axs = plt.subplots(2, 4, figsize=(15, 10))

axs[0, 0].imshow(image_mst, cmap='gray')
axs[0, 0].set_title("Imagem MSTHGR")
axs[0, 0].axis('off')

axs[0, 1].imshow(binary_sauvola_img_mst, cmap='gray')
axs[0, 1].set_title("Imagem MSTHGR Sauvola Threshold")
axs[0, 1].axis('off')

axs[0, 2].imshow(binary_otsu_img_mst, cmap='gray')
axs[0, 2].set_title("Imagem MSTHGR Otsu Threshold")
axs[0, 2].axis('off')

axs[0, 3].imshow(binary_bin_img_mst, cmap='gray')
axs[0, 3].set_title("Imagem MSTHGR Binarization Threshold")
axs[0, 3].axis('off')

# axs[0, 3].imshow(binary_yen_img_mst, cmap='gray')
# axs[0, 3].set_title("Imagem MSTHGR Yen Threshold")
# axs[0, 3].axis('off')

axs[1, 0].imshow(image_org, cmap='gray')
axs[1, 0].set_title("Imagem Original")
axs[1, 0].axis('off')

axs[1, 1].imshow(binary_sauvola_img_org, cmap='gray')
axs[1, 1].set_title("Imagem Original Sauvola Threshold")
axs[1, 1].axis('off')

axs[1, 2].imshow(binary_otsu_img_org, cmap='gray')
axs[1, 2].set_title("Imagem Original Otsu Threshold")
axs[1, 2].axis('off')

axs[1, 3].imshow(binary_bin_img_org, cmap='gray')
axs[1, 3].set_title("Imagem Original Binarization Threshold")
axs[1, 3].axis('off')

# axs[1, 3].imshow(binary_yen_img_org, cmap='gray')
# axs[1, 3].set_title("Imagem Original Yen Threshold")
# axs[1, 3].axis('off')

plt.show()

# image_np = np.array(image_mst)
# fig, ax = try_all_threshold(image_np, figsize=(10, 8), verbose=False)
# plt.show()