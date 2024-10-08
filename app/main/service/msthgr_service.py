import numpy as np
from PIL import Image
from skimage.morphology import white_tophat, black_tophat, disk, reconstruction, erosion, dilation
from skimage import img_as_float

import matplotlib.pyplot as plt
import numpy as np

class MSTHGRService:
    """
    Class to use the Multi-Scale Mathematical Morphology for panoramic dental radiography enhancement.
    """
    FILE_NAME = "msthgr_service.py"

    def __init__(self):
        """
        Initialize the MSTHGRService.
        """
        print("öi")

    def apply_msthgr(self, image: Image.Image, rad=1, iter=9) -> np.ndarray:
        """
        Apply MSTHGR enhancement to the image.
        :param image: The image to be processed as a numpy array.
        :param rad: The initial radius for the structuring element.
        :param iter: The number of iterations to apply.
        :return: The enhanced image in PIL format.
        """
        # print("DEBUG:", self.FILE_NAME, 'apply_msthgr', 'Applying MSTHGR enhancement')

        # Converter para um array NumPy
        ip_arr = np.array(image)

        if ip_arr.ndim == 3 and ip_arr.shape[2] == 3:
            # Converter a imagem RGB para escala de cinza usando a média dos canais
            gray_array = np.mean(ip_arr, axis=2).astype(np.uint8)
        else:
            print("ERROR:", self.FILE_NAME, 'apply_msthgr', 'A entrada deve ser uma imagem RGB com 3 canais')
            return(image)
        
        ip_arr = img_as_float(gray_array)  

        r = rad
        H = disk(r)

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

        RWTH = self.subtract(ip_arr, aReconst)
        dIlation = dilation(ip_arr, footprint=H)
        cReconst = reconstruction(dIlation, ip_arr, method='erosion')
        RBTH = self.subtract(cReconst, ip_arr)

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
                RWTH = self.subtract(ip_arr, aReconst)
                dIlation = dilation(ip_arr, footprint=H)
                cReconst = reconstruction(dIlation, ip_arr, method='erosion')
                RBTH = self.subtract(cReconst, ip_arr)

                RWTH = reconstruction(RWTH, wth, method='dilation')
                RBTH = reconstruction(RBTH, bth, method='dilation')

                matrizWTH.append(RWTH)
                matrizBTH.append(RBTH)

                # Diferença entre iterações
                dRWTH = self.subtract(matrizWTH[k], matrizWTH[k-1])
                matrizDRWTH.append(dRWTH)
                dRBTH = self.subtract(matrizBTH[k], matrizBTH[k-1])
                matrizDRBTH.append(dRBTH)

        m_WTH = matrizWTH[0]
        m_DRWTH = matrizDRWTH[0]
        m_BTH = matrizBTH[0]
        m_DRBTH = matrizDRBTH[0]

        for w in range(1, iter):
            m_WTH = self.max_value_matriz(matrizWTH[w], m_WTH)
            m_BTH = self.max_value_matriz(matrizBTH[w], m_BTH)

        for w in range(1, iter - 1):
            m_DRWTH = self.max_value_matriz(matrizDRWTH[w], m_DRWTH)
            m_DRBTH = self.max_value_matriz(matrizDRBTH[w], m_DRBTH)

        IE = self.enhancement(ip_arr, m_WTH, m_DRWTH, m_BTH, m_DRBTH)

        ip2 = np.clip(IE, 0, 255)

        np_array_normalized = np.clip(ip2, 0, 1) * 255

        # Convert iamge to 8-bits
        np_array_8bit = np_array_normalized.astype(np.uint8)

        # Convert image back to PIL
        im_pil_8bit = Image.fromarray(np_array_8bit)

        return im_pil_8bit
    
    def subtract(self, f1, f2):
        res = f1 - f2
        res[res < 0] = 0

        return res

    def max_value_matriz(self, f1, f2):
        """
        Returns an array where each value is the maximum of the corresponding values ​​of f1 and f2.
        """

        return np.maximum(f1, f2)

    def enhancement(self, f1, f2, f3, f4, f5):
        """
        Performs the improvement calculation based on the addition and subtraction of the provided matrices.
        """
        # Calcula a melhoria
        result = f1 + f2 + f3 - f4 - f5
        # Garantir que os valores fiquem dentro da faixa [0, 255]
        result = np.clip(result, 0, 255)

        return result
    
