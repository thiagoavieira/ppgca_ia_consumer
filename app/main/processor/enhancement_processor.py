import json
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage import filters, img_as_ubyte
from ..processor.abstract_processor import AbstractProcessor
from ..service.msthgr_service import MSTHGRService

class EnhancementProcessor(AbstractProcessor):
    """
    Processor to enhance tooth cropped images with MSTHGR, Gray-scale Conversion, Binarization, Size Adjustment.
    """
    
    FILE_NAME = 'enhancement_processor.py'

    def __init__(self, conf):
        super(EnhancementProcessor, self).__init__(conf)
        self.methods = self.get_config()
        self.__msthgr = MSTHGRService()

    def pre_process(self, input_data: dict) -> dict:
        """
        Pre process the input data
        :param input_data:
        :return:
        """
        print("DEBUG: ", self.FILE_NAME, 'pre_process', 'Checking if the message type is a dict')
        if type(input_data) is not dict:
            input_data = json.loads(input_data)

        return input_data
    
    def process(self, input_data) -> dict:
        """
        Process the input data by applying methods in the specified order.
        :param input_data:
        :return:
        """
        print("DEBUG: Starting processing with methods:", self.methods)
        
        for method in self.methods:
            method_func = getattr(self, method.lower(), None)
            if method_func:
                print(f"DEBUG: Applying method: {method}")
                input_data = method_func(input_data)
            else:
                print(f"ERROR: Method {method} is not implemented.")
        
        return input_data

    def get_config(self) -> list:
        """
        Get configuration from yaml.
        :return:
        """
        
        print("DEBUG: ", self.FILE_NAME, 'get_config', 'Getting config from yaml.')

        if "methods" in self.conf:
            methods = self.conf["methods"]
        else:
            methods = []
            print("ERROR: ", self.FILE_NAME, 'get_config', 'The processor EnhancementProcessor needs configuration.',
                                 self.INTERNAL_SERVER_ERROR)

        return methods
    
    def msthgr(self, input_data: dict) -> dict:
        """
        Apply the MSTHGR enhancement method.
        :param input_data: Dictionary containing the directory of cropped images.
        :return: Updated input_data with processed images.
        """
        cropped_images_directory = input_data.get("cropped_images_directory", "")
        
        # Apply MSTHGR to each image in the directory
        for image_filename in os.listdir(cropped_images_directory):
            if image_filename.endswith(('.jpg', '.png')):
                image_path = os.path.join(cropped_images_directory, image_filename)
                
                # Open the image using PIL and convert to grayscale
                image = Image.open(image_path).convert('L')
                image_array = np.array(image)
                
                enhanced_array = self.__msthgr.apply_msthgr(image_array)
                enhanced_image = Image.fromarray(enhanced_array)
                
                # Save the enhanced image
                enhanced_image_path = os.path.join(cropped_images_directory, f"enhanced_{image_filename}")
                enhanced_image.save(enhanced_image_path)
                
                # cv2.imshow("Original Image", image_array)
                # cv2.imshow("Enhanced Image", np.array(enhanced_image))
                
                # # Wait for a key press and close windows
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            
        return input_data, enhanced_image

    def gray_scale(self, input_data: dict, image) -> dict:
        """
        Convert the image to grayscale.
        """
        return image.convert("L"), input_data
    
    def sauvola_threshold(self, input_data: dict, image) -> dict:
        """
        Applies the Sauvola threshold in the image.
        """
        image_np = np.array(image)
        
        # Convertendo para escala de cinza se não for
        if len(image_np.shape) > 2:
            image_np = self.gray_scale(image_np, input_data)
        
        # Aplicando Sauvola threshold
        thresh_sauvola = filters.threshold_sauvola(image_np)
        binary_sauvola = image_np > thresh_sauvola
        
        # Convertendo de volta para imagem PIL
        binary_sauvola_img = Image.fromarray(img_as_ubyte(binary_sauvola))
        
        return input_data, binary_sauvola_img
    
    def otsu_threshold(self, input_data: dict, image) -> dict:
        """
        Applies the Otsu threshold in the image.
        """
        image_np = np.array(image)
        
        # Convertendo para escala de cinza se não for
        if len(image_np.shape) > 2:
            image_np = self.gray_scale(image_np, input_data)
        
        # Aplicando Otsu threshold
        thresh_otsu = filters.threshold_otsu(image_np)
        binary_otsu = image_np > thresh_otsu
        
        # Convertendo de volta para imagem PIL
        binary_otsu_img = Image.fromarray(img_as_ubyte(binary_otsu))
        
        return input_data, binary_otsu_img

    def binarization(self, input_data: dict, image, threshold=128) -> dict:
        """
        Realiza a binarização da imagem.
        
        :param image: Imagem PIL a ser binarizada.
        :param threshold: Valor de threshold para a binarização (0-255).
        :return: Imagem PIL binarizada.
        """
        binary_image = image.point(lambda p: 255 if p > threshold else 0, mode='1')
    
        return binary_image

    def pad_image_to_max_dimensions(self, image, input_data):
        """
        Padroniza a imagem com a maior altura e largura, adicionando bordas pretas
        para manter a proporção sem cortar a imagem original.
        
        :param image: Imagem PIL a ser processada.
        :param metrics: Dicionário com as métricas 'min_width', 'max_width', 
                        'min_height', 'max_height'.
        :return: Imagem PIL com as dimensões padronizadas.
        """
        max_width = input_data['metrics']['max_width']
        max_height =input_data['metrics']['max_height']
        
        # Redimensiona a imagem para caber dentro das dimensões máximas mantendo a proporção
        image.thumbnail((max_width, max_height), Image.LANCZOS)
        
        # Calcula as bordas necessárias para centralizar a imagem
        delta_width = max_width - image.width
        delta_height = max_height - image.height
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        
        # Adiciona bordas pretas ao redor da imagem para atingir as dimensões máximas
        return input_data, ImageOps.expand(image, padding, fill='black')

    def pad_image_to_average_dimensions(self, image, input_data):
        """
        Redimensiona a imagem para a média das dimensões fornecidas, 
        adicionando bordas pretas se necessário para manter a proporção.
        
        :param image: Imagem PIL a ser processada.
        :param metrics: Dicionário com as métricas 'min_width', 'max_width', 
                        'min_height', 'max_height'.
        :return: Imagem PIL com as dimensões padronizadas.
        """
        avg_width = (input_data['metrics']['min_width'] + input_data['metrics']['max_width']) // 2
        avg_height = (input_data['metrics']['min_height'] + input_data['metrics']['max_height']) // 2
        
        # Redimensiona a imagem para caber dentro das dimensões médias mantendo a proporção
        image.thumbnail((avg_width, avg_height), Image.LANCZOS)
        
        # Calcula as bordas necessárias para centralizar a imagem
        delta_width = avg_width - image.width
        delta_height = avg_height - image.height
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        
        # Adiciona bordas pretas ao redor da imagem para atingir as dimensões médias
        return input_data, ImageOps.expand(image, padding, fill='black')