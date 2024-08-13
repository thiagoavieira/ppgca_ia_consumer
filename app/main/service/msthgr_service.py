from PIL import Image
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_erosion, binary_dilation, grey_dilation, grey_erosion

class MSTHGRService:
    """
    Class to use the Multi-Scale Mathematical Morphology for panoramic dental radiography enhancement.
    """
    FILE_NAME = "msthgr_service.py"

    def __init__(self):
        """
        Initialize the MSTHGRService.
        """
        self.var = "oi"

    def apply_msthgr(self, image_array: np.ndarray, rad=1, iter=7) -> np.ndarray:
        """
        Apply MSTHGR enhancement to the image.
        :param image_array: The image to be processed as a numpy array.
        :param rad: The initial radius for the structuring element.
        :param iter: The number of iterations to apply.
        :return: The enhanced image as a numpy array.
        """
        print("DEBUG: apply_msthgr", "Applying MSTHGR enhancement.")
        
        M, N = image_array.shape
        r = rad
        
        # Convert the image to an 8-bit image if not already
        image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)
        
        def white_top_hat(image, strel):
            return image - binary_erosion(image, structure=strel, mode='constant', cval=0)
        
        def black_top_hat(image, strel):
            return binary_dilation(image, structure=strel, mode='constant', cval=0) - image
        
        def create_strel(radius):
            return np.fromfunction(lambda x, y: (x - radius) ** 2 + (y - radius) ** 2 <= radius ** 2, (2 * radius + 1, 2 * radius + 1), dtype=int)
        
        def reconstruct_by_dilation(erosion_image, original_image, iterations):
            reconstruction = erosion_image.copy()
            for _ in range(iterations):
                reconstruction = binary_dilation(reconstruction, structure=create_strel(r))
            return np.minimum(reconstruction, original_image)
        
        def reconstruct_by_erosion(dilation_image, original_image, iterations):
            reconstruction = dilation_image.copy()
            for _ in range(iterations):
                reconstruction = binary_erosion(reconstruction, structure=create_strel(r))
            return np.maximum(reconstruction, original_image)
        
        def subtract_images(image1, image2):
            return np.clip(image1 - image2, 0, 255)
        
        def max_value_matrix(image1, image2):
            return np.maximum(image1, image2)
        
        # Initialize variables
        H = create_strel(r)
        wth = white_top_hat(image_array, H)
        bth = black_top_hat(image_array, H)
        e_erosion = binary_erosion(image_array, structure=H, mode='constant', cval=0)
        a_reconst = reconstruct_by_dilation(e_erosion, image_array, 8)
        RWTH = subtract_images(image_array, a_reconst)
        d_dilation = binary_dilation(image_array, structure=H, mode='constant', cval=0)
        c_reconst = reconstruct_by_erosion(d_dilation, image_array, 8)
        RBTH = subtract_images(c_reconst, image_array)
        
        RWTH = reconstruct_by_dilation(RWTH, wth, 8)
        RBTH = reconstruct_by_dilation(RBTH, bth, 8)
        
        matriz_wth = [RWTH]
        matriz_bth = [RBTH]
        matriz_drwth = []
        matriz_drbtn = []
        
        d_RWTH = np.zeros((M, N), dtype=np.uint8)
        d_RBTH = np.zeros((M, N), dtype=np.uint8)
        
        if iter > 1:
            for k in range(1, iter):
                r += 1
                H = create_strel(r)
                wth = white_top_hat(image_array, H)
                bth = black_top_hat(image_array, H)
                e_erosion = binary_erosion(image_array, structure=H, mode='constant', cval=0)
                a_reconst = reconstruct_by_dilation(e_erosion, image_array, 8)
                RWTH = subtract_images(image_array, a_reconst)
                d_dilation = binary_dilation(image_array, structure=H, mode='constant', cval=0)
                c_reconst = reconstruct_by_erosion(d_dilation, image_array, 8)
                RBTH = subtract_images(c_reconst, image_array)
                RWTH = reconstruct_by_dilation(RWTH, wth, 8)
                RBTH = reconstruct_by_dilation(RBTH, bth, 8)
                matriz_wth.append(RWTH)
                matriz_bth.append(RBTH)
                d_RWTH = subtract_images(matriz_wth[k], matriz_wth[k-1])
                matriz_drwth.append(d_RWTH)
                d_RBTH = subtract_images(matriz_bth[k], matriz_bth[k-1])
                matriz_drbtn.append(d_RBTH)
        
        # Calculate the maxima
        m_WTH = matriz_wth[0]
        m_DRWTH = matriz_drwth[0]
        m_BTH = matriz_bth[0]
        m_DRBTH = matriz_drbtn[0]
        
        for w in range(1, iter):
            m_WTH = max_value_matrix(matriz_wth[w], m_WTH)
            m_BTH = max_value_matrix(matriz_bth[w], m_BTH)
        for w in range(1, iter - 1):
            m_DRWTH = max_value_matrix(matriz_drwth[w], m_DRWTH)
            m_DRBTH = max_value_matrix(matriz_drbtn[w], m_DRBTH)
        
        # Final enhancement
        enhancement = (image_array + m_WTH + m_DRWTH + m_BTH + m_DRBTH)
        enhancement = np.clip(enhancement, 0, 255).astype(np.uint8)
        
        return enhancement
