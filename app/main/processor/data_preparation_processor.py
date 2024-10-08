import os
import json
import random
import shutil
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
from ..processor.abstract_processor import AbstractProcessor


class DataPreparationProcessor(AbstractProcessor):
    """
    Processor to organize directories to the training process.
    """
    
    FILE_NAME = 'data_preparation_processor.py'

    def __init__(self, conf):
        super(DataPreparationProcessor, self).__init__(conf)
        self.training, self.validation, self.total_images, self.augmentation, self.cross_validation = self.get_configs()
        self._validate_config()

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

    def process(self, input_data: dict) -> dict:
        """
        Process the input by verifying the number of images and dividing them 
        into training and validation directories.
        :param input_data: Dictionary containing the path to cropped images directory
        :return: Updated input_data dictionary
        """
        training_dir = Path(input_data["cropped_images_directory"])
        validation_dir = training_dir.parent.parent / "validation" / training_dir.name

        if validation_dir.exists() and validation_dir.is_dir():
            file_patterns = ["*.jpeg", "*.jpg", "*.png"]
            files_to_delete = []
            for pattern in file_patterns:
                files_to_delete.extend(validation_dir.glob(pattern))
            for file in files_to_delete:
                if file.is_file():
                    try:
                        file.unlink()
                    except Exception as e:
                        print("ERROR: ", self.FILE_NAME, 'process', f"Error deleting {file}: {e}")
        else:
            print("ERROR: ", self.FILE_NAME, 'process', f"Validation directory {validation_dir} does not exist or is not a directory.")

        image_files = list(training_dir.glob("*.*"))
        num_images = len(image_files)

        # Check if the number of images is < total_images
        if num_images < self.total_images and self.augmentation:
            num_augment = min(self.total_images - num_images, num_images)
            augmented_images = self._augment_images(training_dir, num_augment)
            image_files.extend(augmented_images)
            num_images = len(image_files)

        # Check if the number of images exceeds the total_images
        if num_images > self.total_images:
            print(f"INFO: {self.FILE_NAME} process: Exceeding total_images, reducing from {num_images} to {self.total_images}")
            # Randomly delete extra images
            images_to_delete = random.sample(image_files, num_images - self.total_images)
            for image in images_to_delete:
                os.remove(image)
        
        image_files = list(training_dir.glob("*.*"))
        num_images = len(image_files)

        # Calculate number of training and validation images
        num_training = int(num_images * (self.training / 100))
        num_validation = num_images - num_training

        # Randomly shuffle the images
        random.shuffle(image_files)

        # Split images into training and validation sets
        training_images = image_files[:num_training]
        validation_images = image_files[num_training:num_training + num_validation]

        # Ensure the validation directory exists and move validation images to the validation directory
        validation_dir.mkdir(parents=True, exist_ok=True)
        for image in validation_images:
            shutil.move(str(image), validation_dir / image.name)
        input_data["training_images_directory"] = str(training_dir)
        input_data["validation_images_directory"] = str(validation_dir)
        
        return input_data
    
    def _augment_images(self, image_dir: Path, num_augment: int) -> list:
        """
        Applies augmentation to the images in the given directory until the number of 
        augmented images reaches num_augment. Augmented images are saved in the same directory.
        
        :param image_dir: Directory where the original images are stored
        :param num_augment: Number of new augmented images to generate
        :return: List of paths to the augmented images
        """
        augmented_images = []
        image_files = list(image_dir.glob("*.*"))
        
        datagen = ImageDataGenerator(
            rotation_range=20,  # Rotacionar a imagem até 20 graus
            fill_mode='nearest'  # Preencher os pixels que ficam vazios após a rotação
        )

        for image_path in random.sample(image_files, num_augment):
            img = load_img(image_path) 
            x = img_to_array(img)  # Converte a imagem para um array NumPy
            x = np.expand_dims(x, axis=0) 

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=image_dir, save_prefix='aug', save_format='jpeg'):
                augmented_images.append(image_dir / f"aug_{i}.jpeg")
                i += 1
                if i >= 1:
                    break

        return augmented_images
    
    def get_configs(self) -> tuple:
        """
        Get configuration from yaml.
        :return:
        """
        
        print("DEBUG: ", self.FILE_NAME, 'get_configs', 'Getting config from yaml.')
            
        if all(key in self.conf for key in ["training", "validation", "total_images", "cross_validation"]):
            training = self.conf["training"]
            validation = self.conf["validation"]
            total_images = self.conf["total_images"]
            augmentation = self.conf["augmentation"]
            cross_validation = self.conf["cross_validation"]
        else:
            training = 80
            validation = 20
            total_images = 250
            augmentation = False
            cross_validation = False
            print("ERROR: ", self.FILE_NAME, 'get_configs', 'The processor CropProcessor needs configuration, using default configs.',
                                self.INTERNAL_SERVER_ERROR)

        return training, validation, total_images, augmentation, cross_validation
    
    def _validate_config(self):
        """
        Validate training and validation percentages.
        """
        if not (0 <= self.training <= 100):
            raise ValueError(f"Training percentage {self.training} must be between 0 and 100.")
        if not (0 <= self.validation <= 100):
            raise ValueError(f"Validation percentage {self.validation} must be between 0 and 100.")
        if self.training + self.validation != 100:
            raise ValueError(f"Training ({self.training}%) and validation ({self.validation}%) percentages must add up to 100.")


