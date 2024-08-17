import json
import os
import cv2 # type: ignore
import numpy as np # type: ignore
from PIL import Image, ImageOps # type: ignore
from skimage import filters, img_as_ubyte # type: ignore
import concurrent.futures
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
        use_threads = True
        print("DEBUG: Starting processing with methods:", self.methods)
        
        cropped_images_directory = input_data.get("cropped_images_directory", "")
        
        if not cropped_images_directory:
            print("ERROR: No cropped_images_directory specified in input_data.")
            return input_data

        # Listar todas as imagens no diretório
        image_files = [f for f in os.listdir(cropped_images_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        def process_image(image_file, input_data):
            image_path = os.path.join(cropped_images_directory, image_file)
            image = Image.open(image_path)
            
            # Aplicar cada método na imagem
            for method in self.methods:
                method_func = getattr(self, method.lower(), None)
                if method_func:
                    input_data, image = method_func(input_data, image)  # Passa a imagem para o método e recebe a imagem processada de volta
                else:
                    print(f"ERROR: Method {method} is not implemented.")
                    continue
            
            # Sobrescrever a imagem processada no mesmo diretório
            image.save(image_path)
            return image_file, input_data
        
        if use_threads:
            # Processamento paralelo com threads
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_image, image_file, input_data) for image_file in image_files]
                for future in concurrent.futures.as_completed(futures):
                    image_file, input_data = future.result()

        else:
            # Processamento normal em série
            for image_file in image_files:
                _, input_data = process_image(image_file, input_data)
        
        print("INFO: Finished enhancement_processor.py")
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
    
    def msthgr(self, input_data: dict, image: Image.Image) -> tuple:
        """
        Apply the MSTHGR enhancement method.
        :param input_data: Dictionary containing the directory of cropped images.
        :return: Updated input_data with processed images.
        """
        
        msthgr_img = self.__msthgr.apply_msthgr(image)
            
        return input_data, msthgr_img

    def gray_scale(self, input_data: dict, image: Image.Image) -> tuple:
        """
        Convert the image to grayscale.
        """
        return image.convert("L"), input_data
    
    def sauvola_threshold(self, input_data: dict, image: Image.Image) -> tuple:
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
    
    def otsu_threshold(self, input_data: dict, image: Image.Image) -> tuple:
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

    def binarization(self, input_data: dict, image: Image.Image, threshold=128) -> Image.Image:
        """
        Realiza a binarização da imagem.
        
        :param image: Imagem PIL a ser binarizada.
        :param threshold: Valor de threshold para a binarização (0-255).
        :return: Imagem PIL binarizada.
        """
        binary_image = image.point(lambda p: 255 if p > threshold else 0, mode='1')
    
        return input_data, binary_image

    def pad_image_to_max_dimensions(self, input_data: dict, image: Image.Image) -> tuple:
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

    def pad_image_to_average_dimensions(self, input_data: dict, image: Image.Image) -> tuple:
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