import os
import cv2
import numpy as np
from PIL import Image
from skimage import img_as_ubyte
from skimage.filters import threshold_sauvola, threshold_otsu, threshold_yen
from skimage.filters import try_all_threshold
from skimage.exposure import histogram
from PIL import Image, ImageOps 

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

def clahe(input_data: dict, image: Image.Image) -> Image.Image:
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

    return input_data, clahe_image_pil

def pad_image_to_max_dimensions(input_data: dict, image: Image.Image) -> tuple:
    """
    Padroniza a imagem com a maior altura e largura, adicionando bordas pretas
    para manter a proporção sem cortar a imagem original.
    
    :param image: Imagem PIL a ser processada.
    :param metrics: Dicionário com as métricas 'min_width', 'max_width', 
                    'min_height', 'max_height'.
    :return: Imagem PIL com as dimensões padronizadas.
    """
    max_width = 399
    max_height = 436
    
    # Redimensiona a imagem para caber dentro das dimensões máximas mantendo a proporção
    image.thumbnail((max_width, max_height), Image.LANCZOS)
    
    # Calcula as bordas necessárias para centralizar a imagem
    delta_width = max_width - image.width
    delta_height = max_height - image.height
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
    
    # Adiciona bordas pretas ao redor da imagem para atingir as dimensões máximas
    return input_data, ImageOps.expand(image, padding, fill='black')

def pad_image_to_average_dimensions(input_data: dict, image: Image.Image) -> tuple:
    """
    Redimensiona a imagem para a média das dimensões fornecidas, 
    adicionando bordas pretas se necessário para manter a proporção.
    
    :param image: Imagem PIL a ser processada.
    :param metrics: Dicionário com as métricas 'avg_width', 'avg_width'
    :return: Imagem PIL com as dimensões padronizadas.
    """
    avg_width = 206
    avg_height = 261
    
    # Redimensiona a imagem para caber dentro das dimensões médias mantendo a proporção
    image.thumbnail((avg_width, avg_height), Image.LANCZOS)
    
    # Calcula as bordas necessárias para centralizar a imagem
    delta_width = avg_width - image.width
    delta_height = avg_height - image.height
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
    
    # Adiciona bordas pretas ao redor da imagem para atingir as dimensões médias da imagem
    return input_data, ImageOps.expand(image, padding, fill='black')
    

input_data = {}

current_dir = os.path.dirname(os.path.abspath(__file__))
cropped_images_dir = os.path.join(current_dir, '..', 'data', 'cropped_images')
image_mst_path = os.path.join(cropped_images_dir, "97-F-70_block_0_copy.jpg")
image_org_path = os.path.join(cropped_images_dir, "97-F-70_block_0.jpg")

image_mst = Image.open(image_mst_path)
image_org = Image.open(image_org_path)
_, image_clahe = clahe(input_data, image_org)

#_, binary_sauvola_img_mst = sauvola_threshold(input_data, image_mst)
#_, binary_otsu_img_mst = otsu_threshold(input_data, image_mst)
#_, binary_bin_img_mst = binarization(input_data, image_mst)
# _, binary_yen_img_mst = yen_threshold(input_data, image_mst)

#_, binary_sauvola_img_org = sauvola_threshold(input_data, image_org)
#_, binary_otsu_img_org = otsu_threshold(input_data, image_org)
#_, binary_bin_img_org = binarization(input_data, image_org)
# _, binary_yen_img_org = yen_threshold(input_data, image_org)

#fig, axs = plt.subplots(2, 4, figsize=(15, 10))

#axs[0, 0].imshow(image_mst, cmap='gray')
#axs[0, 0].set_title("Imagem MSTHGR")
#axs[0, 0].axis('off')

#axs[0, 1].imshow(binary_sauvola_img_mst, cmap='gray')
#axs[0, 1].set_title("Imagem MSTHGR Sauvola Threshold")
#axs[0, 1].axis('off')

#axs[0, 2].imshow(binary_otsu_img_mst, cmap='gray')
#axs[0, 2].set_title("Imagem MSTHGR Otsu Threshold")
#axs[0, 2].axis('off')

#axs[0, 3].imshow(binary_bin_img_mst, cmap='gray')
#axs[0, 3].set_title("Imagem MSTHGR Binarization Threshold")
#axs[0, 3].axis('off')

# axs[0, 3].imshow(binary_yen_img_mst, cmap='gray')
# axs[0, 3].set_title("Imagem MSTHGR Yen Threshold")
# axs[0, 3].axis('off')

#axs[1, 0].imshow(image_org, cmap='gray')
#axs[1, 0].set_title("Imagem Original")
#axs[1, 0].axis('off')

#axs[1, 1].imshow(binary_sauvola_img_org, cmap='gray')
#axs[1, 1].set_title("Imagem Original Sauvola Threshold")
#axs[1, 1].axis('off')

#axs[1, 2].imshow(binary_otsu_img_org, cmap='gray')
#axs[1, 2].set_title("Imagem Original Otsu Threshold")
#axs[1, 2].axis('off')

# axs[1, 3].imshow(binary_bin_img_org, cmap='gray')
# axs[1, 3].set_title("Imagem Original Binarization Threshold")
# axs[1, 3].axis('off')

# axs[1, 3].imshow(binary_yen_img_org, cmap='gray')
# axs[1, 3].set_title("Imagem Original Yen Threshold")
# axs[1, 3].axis('off')

#plt.show()

# image_np = np.array(image_mst)
# fig, ax = try_all_threshold(image_np, figsize=(10, 8), verbose=False)
# plt.show()

_, image_clahe_msthgr = clahe(input_data, image_mst)

_, binary_sauvola_img_org = pad_image_to_max_dimensions(input_data, image_org)
_, binary_otsu_img_org = pad_image_to_average_dimensions(input_data, image_org)
#_, binary_bin_img_org = binarization(input_data, image_org)
# _, binary_yen_img_org = yen_threshold(input_data, image_org)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(image_mst, cmap='gray')
axs[0, 0].set_title("Imagem MSTHGR")
axs[0, 0].axis('off')

axs[0, 1].imshow(image_clahe, cmap='gray')
axs[0, 1].set_title("Imagem CLAHE")
axs[0, 1].axis('off')

axs[0, 2].imshow(image_clahe_msthgr, cmap='gray')
axs[0, 2].set_title("Imagem CLAHE MSTHGR")
axs[0, 2].axis('off')

#axs[0, 3].imshow(image_msthgr_clahe, cmap='gray')
#axs[0, 3].set_title("Imagem MSTHGR CLAHE")
#axs[0, 3].axis('off')

# axs[0, 3].imshow(binary_yen_img_mst, cmap='gray')
# axs[0, 3].set_title("Imagem MSTHGR Yen Threshold")
# axs[0, 3].axis('off')

axs[1, 0].imshow(image_org, cmap='gray')
axs[1, 0].set_title("Imagem Original")
axs[1, 0].axis('off')

axs[1, 1].imshow(binary_sauvola_img_org, cmap='gray')
axs[1, 1].set_title("Imagem Original Pad Max")
axs[1, 1].axis('off')

axs[1, 2].imshow(binary_otsu_img_org, cmap='gray')
axs[1, 2].set_title("Imagem Original Pad Avg")
axs[1, 2].axis('off')

# axs[1, 3].imshow(binary_bin_img_org, cmap='gray')
# axs[1, 3].set_title("Imagem Original Binarization Threshold")
# axs[1, 3].axis('off')

# axs[1, 3].imshow(binary_yen_img_org, cmap='gray')
# axs[1, 3].set_title("Imagem Original Yen Threshold")
# axs[1, 3].axis('off')

plt.show()