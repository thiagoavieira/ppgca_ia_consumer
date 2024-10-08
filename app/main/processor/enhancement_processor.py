import json
import os
import cv2 # type: ignore
import numpy as np # type: ignore
from PIL import Image, ImageOps # type: ignore
from skimage import img_as_ubyte
from skimage.filters import threshold_sauvola, threshold_otsu
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
        print("DEBUG: ", self.FILE_NAME, 'process', f"Starting processing with methods: {self.methods}")
        
        cropped_images_directory = input_data.get("cropped_images_directory", "")
        
        if not cropped_images_directory:
            print("ERROR: ", self.FILE_NAME, 'process', 'No cropped_images_directory specified in input_data.')
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
                    print("ERROR: ", self.FILE_NAME, 'process', f'Method {method} is not implemented."')
                    continue
            
            # Sobrescrita da imagem
            image.save(image_path)
            return image_file, input_data
        
        if use_threads:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_image, image_file, input_data) for image_file in image_files]
                for future in concurrent.futures.as_completed(futures):
                    image_file, input_data = future.result()
        else:
            for image_file in image_files:
                _, input_data = process_image(image_file, input_data)
        
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
        Convert the image to grayscale and return it as a PIL Image.
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("L"))
        else:
            if len(image.shape) > 2:
                image_np = np.mean(image, axis=2).astype(np.uint8)
            else:
                image_np = image

        grayscale_image_pil = Image.fromarray(image_np)

        return input_data, grayscale_image_pil

    def sauvola_threshold(self, input_data: dict, image: Image.Image) -> tuple:
        """
        Applies the Sauvola threshold in the image.
        """
        input_data, image_np = self.gray_scale(input_data, image)
        
        thresh_sauvola = threshold_sauvola(image_np, window_size=25, k=0.15)
        binary_sauvola = image_np > thresh_sauvola
        
        binary_sauvola_img = Image.fromarray(img_as_ubyte(binary_sauvola))

        return input_data, binary_sauvola_img

    def otsu_threshold(self, input_data: dict, image: Image.Image) -> tuple:
        """
        Applies the Otsu threshold in the image.
        """
        input_data, image_np = self.gray_scale(input_data, image)
        
        thresh_otsu = threshold_otsu(image_np, nbins=256)
        binary_otsu = image_np > thresh_otsu
        
        binary_otsu_img = Image.fromarray(img_as_ubyte(binary_otsu))
        
        return input_data, binary_otsu_img
    
    def clahe(self, input_data: dict, image: Image.Image) -> Image.Image:
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

    def binarization(self, input_data: dict, image: Image.Image, threshold=128) -> Image.Image:
        """
        Realiza a binarização da imagem.
        
        :param image: Imagem PIL a ser binarizada.
        :param threshold: Valor de threshold para a binarização (0-255).
        :return: Imagem PIL binarizada.
        """
        input_data, image_np = self.gray_scale(input_data, image)
    
        binary_np = np.where(image_np > threshold, 255, 0).astype(np.uint8)
        
        binary_image = Image.fromarray(binary_np)
        
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
        :param metrics: Dicionário com as métricas 'avg_width', 'avg_width'
        :return: Imagem PIL com as dimensões padronizadas.
        """
        avg_width = input_data['metrics']['avg_width']
        avg_height = input_data['metrics']['avg_height']
        
        # Redimensiona a imagem para caber dentro das dimensões médias mantendo a proporção
        image.thumbnail((avg_width, avg_height), Image.LANCZOS)
        
        # Calcula as bordas necessárias para centralizar a imagem
        delta_width = avg_width - image.width
        delta_height = avg_height - image.height
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        
        # Adiciona bordas pretas ao redor da imagem para atingir as dimensões médias da imagem
        return input_data, ImageOps.expand(image, padding, fill='black')
    
