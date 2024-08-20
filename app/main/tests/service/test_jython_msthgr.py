import os
import sys
import imagej
import scyjava
from scyjava import jimport
import numpy as np
from PIL import Image
from skimage import morphology
from skimage.morphology import white_tophat, black_tophat, disk, reconstruction, erosion, dilation
from skimage import io, img_as_float, img_as_ubyte

import matplotlib.pyplot as plt
import numpy as np


def display_image(image_pil):
    plt.figure(figsize=(8, 8))
    plt.imshow(image_pil, cmap='gray')
    plt.axis('off')
    plt.show()


def normalize_image(image):
    """ Normaliza a imagem para o intervalo [0, 1] para imagens em float. """
    if image.dtype == np.float32 or image.dtype == np.float64:
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val)  # Normaliza para [0, 1]
    return image

def process_and_convert_image(image):
    """ Processa e converte a imagem para uint8. """
    # Normalize a imagem se necessário
    image = normalize_image(image)
    
    # Converte a imagem normalizada para uint8
    return img_as_ubyte(image)

# Increasing the memory available to Java
scyjava.config.add_options('-Xmx6g')
ij = imagej.init(add_legacy=True, mode='interactive')

def subtract(f1, f2):
    res = f1 - f2
    res[res < 0] = 0
    return res

def max_value_matriz(f1, f2):
    """Retorna uma matriz onde cada valor é o máximo entre os valores correspondentes de f1 e f2."""
    return np.maximum(f1, f2)

def enhancement(f1, f2, f3, f4, f5):
    """Realiza o cálculo de melhoria baseado na soma e subtração das matrizes fornecidas."""
    # Calcular a melhoria
    result = f1 + f2 + f3 - f4 - f5
    # Garantir que os valores fiquem dentro da faixa [0, 255]
    result = np.clip(result, 0, 255)
    return result

def image_processor_to_numpy(ip):
    width = ip.getWidth()
    height = ip.getHeight()
    np_array = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            np_array[y, x] = ip.getPixel(x, y)
    
    return img_as_float(np_array)

def MSTHGR(path: str, rad: int, iter: int) -> Image.Image:
    
    # imp = ij.io().open(path)
    # imp = ij.py.to_imageplus(imp)
    # imp.getProcessor().convertToByte(True)
    # imp2 = imp.duplicate()
    # M = imp2.getWidth()
    # N = imp2.getHeight()
    # ip = imp2.getProcessor()
    
    imp = ij.io().open(path)
    imp = ij.py.to_imageplus(imp)
    ip = imp.getProcessor()

    # Converter para numpy array
    ip_arr = image_processor_to_numpy(ip)

    r = rad
    print(ip_arr)
    H = disk(r)  # Elemento estruturante

    matrizWTH = []
    matrizBTH = []
    matrizDRWTH = []
    matrizDRBTH = []

    # Operações morfológicas
    wth = white_tophat(ip_arr, footprint=H)
    bth = black_tophat(ip_arr, footprint=H)
    eRosion = erosion(ip_arr, footprint=H)
    aReconst = reconstruction(eRosion, ip_arr, method='dilation')

    RWTH = subtract(ip_arr, aReconst)
    dIlation = dilation(ip_arr, footprint=H)
    cReconst = reconstruction(dIlation, ip_arr, method='erosion')
    RBTH = subtract(cReconst, ip_arr)

    RWTH = reconstruction(RWTH, wth, method='dilation')
    RBTH = reconstruction(RBTH, bth, method='dilation')

    matrizWTH.append(RWTH)
    matrizBTH.append(RBTH)

    # Iterações adicionais
    if iter > 1:
        for k in range(1, iter):
            r += 1
            H = disk(r)
            wth = white_tophat(ip_arr, footprint=H)
            bth = black_tophat(ip_arr, footprint=H)
            eRosion = erosion(ip_arr, footprint=H)
            aReconst = reconstruction(eRosion, ip_arr, method='dilation')
            RWTH = subtract(ip_arr, aReconst)
            dIlation = dilation(ip_arr, footprint=H)
            cReconst = reconstruction(dIlation, ip_arr, method='erosion')
            RBTH = subtract(cReconst, ip_arr)

            RWTH = reconstruction(RWTH, wth, method='dilation')
            RBTH = reconstruction(RBTH, bth, method='dilation')

            matrizWTH.append(RWTH)
            matrizBTH.append(RBTH)

            # Diferença entre iterações
            dRWTH = subtract(matrizWTH[k], matrizWTH[k-1])
            matrizDRWTH.append(dRWTH)
            dRBTH = subtract(matrizBTH[k], matrizBTH[k-1])
            matrizDRBTH.append(dRBTH)

    m_WTH = matrizWTH[0]
    m_DRWTH = matrizDRWTH[0]
    m_BTH = matrizBTH[0]
    m_DRBTH = matrizDRBTH[0]

    for w in range(1, iter):
        m_WTH = max_value_matriz(matrizWTH[w], m_WTH)
        m_BTH = max_value_matriz(matrizBTH[w], m_BTH)

    for w in range(1, iter - 1):
        m_DRWTH = max_value_matriz(matrizDRWTH[w], m_DRWTH)
        m_DRBTH = max_value_matriz(matrizDRBTH[w], m_DRBTH)

    IE = enhancement(ip_arr, m_WTH, m_DRWTH, m_BTH, m_DRBTH)

    IE = np.clip(IE, 0, 255)

    # plt.figure(figsize=(8, 8))
    # plt.imshow(IE, cmap='gray')
    # plt.axis('off')
    # plt.show()

    return IE


print(type(ij))
print(f"ÏmageJ version: {ij.getVersion()}")

Runtime = jimport('java.lang.Runtime')
print(Runtime.getRuntime().maxMemory() // (2**20), " MB available to Java")

current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, '..', 'data', 'images')
cropped_images_dir = os.path.join(current_dir, '..', 'data', 'cropped_images')
path = os.path.join(cropped_images_dir, "97-F-70_block_0.jpg")
full_image_path = os.path.join(images_dir, "31-M-51.jpg.jpg")

ij_obj = ij.io().open(path)
imp = ij.py.to_imageplus(ij_obj)

imp2 = imp.duplicate()

M = imp2.getWidth()
N = imp2.getHeight()
print("M: ", M, ", N: ", N)
ip = imp2.getProcessor()
print("ip: ", ip)

imageO = np.zeros((M, N), dtype=int)

for j in range(N):
    for i in range(M):
        p = ip.getPixel(i, j)
        imageO[i, j] = p

image = np.zeros((M, N), dtype=int)
rad = 1
iter = 9


def numpy_to_pil(np_array):
    return Image.fromarray(np_array)

ip2 = MSTHGR(path, rad, iter)

np_array_normalized = np.clip(ip2, 0, 1) * 255

# Converter para 8 bits
np_array_8bit = np_array_normalized.astype(np.uint8)

# Converter de volta para imagem PIL
im_pil_8bit = Image.fromarray(np_array_8bit)
resolucao = im_pil_8bit.size

# Imprimindo a resolução
print("Resolução da imagem:", resolucao)
display_image(im_pil_8bit)
