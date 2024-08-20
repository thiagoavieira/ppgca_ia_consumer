import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import white_tophat, black_tophat, disk, reconstruction, erosion, dilation
from skimage import img_as_float


def display_image(image_pil):
    plt.figure(figsize=(8, 8))
    plt.imshow(image_pil, cmap='gray')
    plt.axis('off')
    plt.show()


def subtract(f1, f2):
    res = f1 - f2
    res[res < 0] = 0
    return res

def max_value_matriz(f1, f2):
    """
    Retorna uma matriz onde cada valor é o máximo entre os valores correspondentes de f1 e f2.
    """
    return np.maximum(f1, f2)

def enhancement(f1, f2, f3, f4, f5):
    """
    Realiza o cálculo de melhoria baseado na soma e subtração das matrizes fornecidas.
    """
    # Calcula a melhoria
    result = f1 + f2 + f3 - f4 - f5
    # Garante que os valores fiquem dentro da faixa [0, 255]
    result = np.clip(result, 0, 255)
    return result


def MSTHGR(path: str, rad: int, iter: int) -> Image.Image: 
    im_pil = Image.open(path)
    ip_arr = np.array(im_pil)

    if ip_arr.ndim == 3 and ip_arr.shape[2] == 3:
        # Converter a imagem RGB para escala de cinza usando a média dos canais
        gray_array = np.mean(ip_arr, axis=2).astype(np.uint8)
    else:
        raise ValueError("A entrada deve ser uma imagem RGB com 3 canais")
    
    ip_arr = img_as_float(gray_array)  

    r = rad
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

    return IE

current_dir = os.path.dirname(os.path.abspath(__file__))
cropped_images_dir = os.path.join(current_dir, '..', 'data', 'cropped_images')
path = os.path.join(cropped_images_dir, "97-F-70_block_0.jpg")

rad = 1
iter = 9

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
