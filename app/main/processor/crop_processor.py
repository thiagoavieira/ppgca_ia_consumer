import json
from PIL import Image
import os

from ..processor.abstract_processor import AbstractProcessor


class CropProcessor(AbstractProcessor):
    """
    Processor to crop images according to annotations.
    """
    
    FILE_NAME = 'crop_processor.py'

    def __init__(self, conf):
        super(CropProcessor, self).__init__(conf)
        self.save_mode = self.get_config()

    def pre_process(self, input_data: dict) -> dict:
        """
        Pre process the input data
        :param input_data: The input data as a dict
        :return: Processed input data as a formatted dictionary
        """
        print("DEBUG:", self.FILE_NAME, 'pre_process', 'Checking if the message type is a dict')
        
        if isinstance(input_data, dict):
            (description, teeth_type_list) = next(iter(input_data.items()))

            output = {
                'teeth_type_list': teeth_type_list,
                'description': description,
                'save_mode': True
            }
            return output

        # Se input_data não for um dict, tenta carregar como JSON
        try:
            output = json.loads(input_data)
            return output
        except json.JSONDecodeError:
            print("ERROR:", self.FILE_NAME, 'pre_process', 'Failed to decode JSON')
            return {}

    def process(self, input_data: dict) -> dict:
        """
        Process the input checking if the message will be cropped with save mode or not.
        :param input_data:
        :return:
        """

        current_directory = os.path.dirname(os.path.abspath(__file__))

        cropped_images_directory = os.path.normpath(os.path.join(current_directory, '..', 'data', 'cropped_images', 'training'))
        images_directory = os.path.normpath(os.path.join(current_directory, '..', 'data', 'images'))
        annotations_directory = os.path.normpath(os.path.join(current_directory, '..', 'data', 'annotations'))

        description = input_data.get('description', "no_name")
        self.__prepare_directory(cropped_images_directory, description)
        
        # Variáveis para armazenar métricas
        min_width, max_width = float('inf'), 0
        min_height, max_height = float('inf'), 0

        for filename in os.listdir(annotations_directory):
            if filename.endswith('.json'):
                annotation_path = os.path.join(annotations_directory, filename)
                annotations = self.__load_annotations(annotation_path)
                
                filtered_blocks = self.__filter_annotations_by_teeth_type(annotations, input_data.get('teeth_type_list', []))
                
                if filtered_blocks:
                    image_filename = filename.replace('.json', '.jpg')
                    image_path = os.path.join(images_directory, image_filename)
                    
                    if not os.path.exists(image_path):
                        image_filename = filename.replace('.json', '.png')
                        image_path = os.path.join(images_directory, image_filename)
                    
                    if os.path.exists(image_path):
                        metrics = self.__crop_and_save_images(image_path, filtered_blocks, cropped_images_directory, description)

                        # Atualiza as métricas
                        min_width = min(min_width, metrics['min_width'])
                        max_width = max(max_width, metrics['max_width'])
                        min_height = min(min_height, metrics['min_height'])
                        max_height = max(max_height, metrics['max_height'])

        print("INFO:", self.FILE_NAME, '__crop_and_save_images', f'All cropped images saved from class: {description}')

        input_data['cropped_images_directory'] = os.path.join(cropped_images_directory, description)
        input_data['metrics'] = {
            'min_width': min_width,
            'max_width': max_width,
            'min_height': min_height,
            'max_height': max_height
        }
        print("INFO:", self.FILE_NAME, '__crop_and_save_images', 'Size metrics: {}'.format(input_data.get('metrics')))
        return input_data
    
    def __load_annotations(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
        
    def __filter_annotations_by_teeth_type(self, annotations, teeth_type_list):
        filtered_blocks = []
        for annotation in annotations['annotations']:
            for block in annotation['blocks']:
                if any(state in teeth_type_list for state in block['states']):
                    filtered_blocks.append(block)
        return filtered_blocks
    
    def __prepare_directory(self, base_directory, description):
        description_directory = os.path.join(base_directory, description)
        
        if os.path.exists(description_directory):
            print("INFO:", self.FILE_NAME, '__prepare_directory', f'Deleting existing images from class: {description}')
            for filename in os.listdir(description_directory):
                file_path = os.path.join(description_directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            print("INFO:", self.FILE_NAME, '__prepare_directory', f'Creating new directory for class: {description}')
            os.makedirs(description_directory)
        
    def __crop_and_save_images(self, image_path, blocks, cropped_images_directory, description):
        description_directory = os.path.join(cropped_images_directory, description)
        os.makedirs(description_directory, exist_ok=True)
        
        image = Image.open(image_path)
        
        # Variáveis para armazenar métricas
        min_width, max_width = float('inf'), 0
        min_height, max_height = float('inf'), 0
        
        for i, block in enumerate(blocks):
            points = block['points']
            x_min = min(p['x'] for p in points)
            y_min = min(p['y'] for p in points)
            x_max = max(p['x'] for p in points)
            y_max = max(p['y'] for p in points)
            
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_image_filename = f"{os.path.basename(image_path).split('.')[0]}_block_{i}.jpg"
            cropped_image_path = os.path.join(description_directory, cropped_image_filename)
            
            cropped_image.save(cropped_image_path)
            
            # Atualiza as métricas
            width, height = cropped_image.size
            min_width = min(min_width, width)
            max_width = max(max_width, width)
            min_height = min(min_height, height)
            max_height = max(max_height, height)
        
        return {
            'min_width': min_width,
            'max_width': max_width,
            'min_height': min_height,
            'max_height': max_height
        }
    
    def get_config(self) -> tuple:
        """
        Get configuration from yaml.
        :return:
        """
        
        print("DEBUG: ", self.FILE_NAME, 'get_config', 'Getting config from yaml.')

        if "save_mode" in self.conf:
            save_mode = self.conf["save_mode"]
        else:
            save_mode = False
            print("ERROR: ", self.FILE_NAME, 'get_config', 'The processor CropProcessor needs configuration.',
                                 self.INTERNAL_SERVER_ERROR)

        return save_mode

