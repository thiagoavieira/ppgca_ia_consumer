import json
import numpy as np
import torch
import time
import os
import tensorflow as tf
from datetime import datetime
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
# For VGG16 and GoogleNet
#from keras.models import Model, Sequential
#from keras.layers import Input, Dense, Conv2D, MaxPool2D , Flatten, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dropout
#from keras.optimizers import Adam
#from keras.callbacks import ModelCheckpoint, EarlyStopping
#from keras.src.legacy.preprocessing.image import ImageDataGenerator
# For ResNet50V2
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, concatenate, Input, Dense, Conv2D, MaxPool2D, Flatten, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from ..processor.abstract_processor import AbstractProcessor


class CNNTrainingProcessor(AbstractProcessor):
    """
    Processor to organize directories to the training process.
    """
    
    FILE_NAME = 'cnn_training_processor.py'

    def __init__(self, conf):
        super(CNNTrainingProcessor, self).__init__(conf)
        self.networks, self.activations, self.lr, self.epochs, self.width_height, self.num_classes = self.get_configs()
        self.gpu_count = torch.cuda.device_count()
        self.device_queue = list(range(self.gpu_count)) if self.gpu_count > 0 else ['cpu']
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.training_path = os.path.join(self.current_directory, '..', 'data', 'cropped_images', 'training')
        self.validation_path = os.path.join(self.current_directory, '..', 'data', 'cropped_images', 'validation')

    def pre_process(self, input_data: dict) -> dict:
        """
        Pre process the input data
        :param input_data:
        :return:
        """
        print("DEBUG: ", self.FILE_NAME, 'pre_process', 'Checking if the message type is a dict')
        if type(input_data) is not dict:
            input_data = json.loads(input_data)

        if isinstance(self.width_height, str):
            if self.width_height == 'pad_image_to_max_dimensions':
                self.width, self.height = self.get_max_dimensions()
            elif self.width_height == 'pad_image_to_average_dimensions':
                self.width, self.height = self.get_average_dimensions()
            else:
                self.width, self.height = map(int, self.width_height.split(','))

        return input_data

    def process(self, input_data: dict) -> dict:
        """
        Process the input data
        :param input_data:
        :return:
        """
        if self.gpu_count == 0:
            print("INFO: ", self.FILE_NAME, 'process', "No GPU available, processing on CPU.")
            for network in self.networks:
                input_data = self._process_on_device(input_data, network, 'cpu')
        else:
            for network in self.networks:
                device = self._get_next_available_device()
                print("INFO: ", self.FILE_NAME, 'process', f"GPU available, processing on GPU: {device}")
                input_data = self._process_on_device(input_data, network, device)

        return input_data
    
    def _get_next_available_device(self):
        """
        Get the next available GPU or CPU for processing. Wait if all GPUs are busy.
        :return: The ID of the available device (GPU or CPU).
        """
        while not self.device_queue:
            time.sleep(1)  # Wait for a device to become available

        return self.device_queue.pop(0)
    
    def _process_on_device(self, input_data: dict, network: str, device: str) -> dict:
        """
        Process the input data using the specified network on the given device.
        :param input_data: entry's dict.
        :param network: CNN's name.
        :param device: GPU or CPU where the processing occurs.
        :return: final's dict.
        """
        print("INFO: ", self.FILE_NAME, '_process_on_device', f"Processing {network} on {device}")

        if network == 'VGG16':
            input_data = self._process_vgg16(input_data, device)
        elif network == 'ResNet50V2':
            input_data = self._process_resnet50v2(input_data, device)
        elif network == 'GoogleNet':
            input_data = self._process_googlenet(input_data, device)
        else:
            print("DEBUG: ", self.FILE_NAME, '_process_on_device', f"WARNING: Rede {network} não é suportada.")

        self.device_queue.append(device)  # Mark the device as available again
        return input_data

    def _process_vgg16(self, input_data: dict, device: str) -> dict:
        print("INFO: ", self.FILE_NAME, '_process_vgg16', f"Running VGG16 on {device}")

        trdata = ImageDataGenerator()
        traindata = trdata.flow_from_directory(
            directory=self.training_path,
            target_size=(self.width,self.height),
            batch_size=32,
            class_mode='categorical'
        )
        tsdata = ImageDataGenerator()
        testdata = tsdata.flow_from_directory(
            directory=self.validation_path,
            target_size=(self.width,self.height),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        base_model = tf.keras.applications.VGG16(
            input_shape=(self.width, self.height, self.num_classes), 
            include_top=False,  # remove a parte totalmente conectada
            weights='imagenet'
        )
        base_model.trainable = True

        tuning_layer_name = 'block4_pool' 
        tuning_layer = base_model.get_layer(tuning_layer_name)
        tuning_index = base_model.layers.index(tuning_layer)

        # Congela as camadas ate o tuning_index
        for layer in base_model.layers[:tuning_index]:
            layer.trainable = False

        base_model.summary()

        data_transforms = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./127.5, offset=-1)
        ], name='data_transforms')

        model = tf.keras.Sequential([
            data_transforms,
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        learning_rate = self.lr
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "vgg16_best_model.keras", 
            monitor='val_accuracy', 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='max', 
            save_freq='epoch'
        )

        early = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            min_delta=0, 
            patience=50, 
            verbose=1, 
            mode='max'
        )

        hist = model.fit(
            traindata, 
            validation_data=testdata, 
            epochs=self.epochs, 
            callbacks=[checkpoint, early]
        )

        results_directory = os.path.join(self.current_directory, '..', 'data', 'results', 'VGG16')
        os.makedirs(results_directory, exist_ok=True)
        final_accuracy = hist.history['accuracy'][-1]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")

        model_filename = f"{final_accuracy:.2f}_{timestamp}.h5"
        model_filepath = os.path.join(results_directory, model_filename)

        model.save(model_filepath)

        print(f"Modelo salvo em: {model_filepath}")

        y_true = testdata.classes  # Os verdadeiros do conjunto de validação
        y_score = model.predict(testdata)  # As probabilidades preditas pelo modelo
        y_pred = np.argmax(y_score, axis=1)  # As classes preditas com base nas probabilidades
        f1 = f1_score(y_true, y_pred, average='weighted')

        input_data['VGG16_output'] = hist.history
        input_data['VGG16_model'] = f"{final_accuracy:.2f}_{timestamp}"
        input_data['VGG16_y_true'] = y_true.tolist()
        input_data['VGG16_y_score'] = y_score.tolist() 
        input_data['VGG16_y_pred'] = y_pred.tolist()
        input_data['VGG16_f1_score'] = f1

        return input_data

    def _process_resnet50v2(self, input_data: dict, device: str) -> dict:
        print("INFO: ", self.FILE_NAME, '_process_resnet50v2',f"Running ResNet50V2 on {device}")

        trdata = ImageDataGenerator()
        traindata = trdata.flow_from_directory(
            directory=self.training_path,
            target_size=(self.width,self.height),
            batch_size=32,
            class_mode='categorical'
        )
        tsdata = ImageDataGenerator()
        testdata = tsdata.flow_from_directory(
            directory=self.validation_path,
            target_size=(self.width,self.height),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        base_model = tf.keras.applications.ResNet50V2(input_shape=(self.width,self.height,self.num_classes), include_top=False)
        base_model.trainable = True

        tuning_layer_name = 'conv5_block1_preact_bn'
        tuning_layer = base_model.get_layer(tuning_layer_name)
        tuning_index = base_model.layers.index(tuning_layer)

        for layer in base_model.layers[:tuning_index]:
            layer.trainable = False

        base_model.summary()

        data_transforms = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./127.5, offset=-1)
        ], name='data_transforms')

        model = tf.keras.Sequential([
            data_transforms, 
            base_model, 
            tf.keras.layers.GlobalAveragePooling2D(), 
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        learning_rate = self.lr
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "resnet50v2_best_model.keras", 
            monitor='val_accuracy', 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='max', 
            save_freq='epoch'
        )

        early = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            min_delta=0, 
            patience=50, 
            verbose=1, 
            mode='max'
        )

        hist = model.fit(
            traindata, 
            validation_data=testdata, 
            epochs=self.epochs, 
            callbacks=[checkpoint, early]
        )

        results_directory = os.path.join(self.current_directory, '..', 'data', 'results', 'ResNet50v2')
        os.makedirs(results_directory, exist_ok=True)
        final_accuracy = hist.history['accuracy'][-1]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")

        model_filename = f"{final_accuracy:.2f}_{timestamp}.h5"
        model_filepath = os.path.join(results_directory, model_filename)
        
        model.save(model_filepath)
        print(f"Modelo salvo em: {model_filepath}")

        y_true = testdata.classes  # Os valores verdadeiros do conjunto de validação
        y_score = model.predict(testdata)  # As probabilidades preditas pelo modelo
        y_pred = np.argmax(y_score, axis=1)  # As classes preditas com base nas probabilidades
        f1 = f1_score(y_true, y_pred, average='weighted')

        input_data['ResNet50V2_output'] = hist.history
        input_data['ResNet50V2_model'] = f"{final_accuracy:.2f}_{timestamp}"
        input_data['ResNet50V2_y_true'] = y_true.tolist()
        input_data['ResNet50V2_y_score'] = y_score.tolist() 
        input_data['ResNet50V2_y_pred'] = y_pred.tolist()
        input_data['ResNet50V2_f1_score'] = f1

        return input_data

    def _process_googlenet(self, input_data: dict, device: str) -> dict:
        print("INFO: ", self.FILE_NAME, '_process_googlenet',f"Running GoogleNet on {device}")

        trdata = ImageDataGenerator()
        traindata = trdata.flow_from_directory(
            directory=self.training_path,
            target_size=(self.width,self.height),
            batch_size=32,
            class_mode='categorical'
        )
        tsdata = ImageDataGenerator()
        testdata = tsdata.flow_from_directory(
            directory=self.validation_path,
            target_size=(self.width,self.height),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        base_model = tf.keras.applications.InceptionV3(
            input_shape=(self.width, self.height, self.num_classes), 
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = True

        tuning_layer_name = 'mixed7'  # Exemplo de camada na InceptionV3 e onde o finetunning vai comecar
        tuning_layer = base_model.get_layer(tuning_layer_name)
        tuning_index = base_model.layers.index(tuning_layer)

        for layer in base_model.layers[:tuning_index]:
            layer.trainable = False

        base_model.summary()

        data_transforms = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./127.5, offset=-1)
        ], name='data_transforms')

        model = tf.keras.Sequential([
            data_transforms,
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        learning_rate = self.lr
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "googlenet_best_model.keras", 
            monitor='val_accuracy', 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='max', 
            save_freq='epoch'
        )

        early = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            min_delta=0, 
            patience=20, 
            verbose=1, 
            mode='max'
        )

        hist = model.fit(
            traindata, 
            validation_data=testdata, 
            epochs=self.epochs, 
            callbacks=[checkpoint, early]
        )

        results_directory = os.path.join(self.current_directory, '..', 'data', 'results', 'GoogLeNet')
        os.makedirs(results_directory, exist_ok=True)
        final_accuracy = hist.history['accuracy'][-1]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")

        model_filename = f"{final_accuracy:.2f}_{timestamp}.h5"
        model_filepath = os.path.join(results_directory, model_filename)
        
        model.save(model_filepath)
        print(f"Modelo salvo em: {model_filepath}")

        y_true = testdata.classes  # Os verdadeiros do conjunto de validação
        y_score = model.predict(testdata)  # As probabilidades preditas pelo modelo
        y_pred = np.argmax(y_score, axis=1)  # As classes preditas com base nas probabilidades
        f1 = f1_score(y_true, y_pred, average='weighted')

        input_data['GoogLeNet_output'] = hist.history
        input_data['GoogLeNet_model'] = f"{final_accuracy:.2f}_{timestamp}"
        input_data['GoogLeNet_y_true'] = y_true.tolist()
        input_data['GoogLeNet_y_score'] = y_score.tolist() 
        input_data['GoogLeNet_y_pred'] = y_pred.tolist()
        input_data['GoogLeNet_f1_score'] = f1
        
        return input_data
    
    def get_max_dimensions(self):
        """
        Get the maximum width and height of images in the training and validation directories.
        """
        image_paths = self.get_image_paths()
        max_width = 0
        max_height = 0
        
        with ThreadPoolExecutor() as executor:
            future_to_image = {executor.submit(self.process_image, img_path): img_path for img_path in image_paths}
            
            for future in as_completed(future_to_image):
                width, height = future.result()
                max_width = max(max_width, width)
                max_height = max(max_height, height)
        
        return max_width, max_height

    def get_average_dimensions(self):
        """
        Get the average width and height of images in the training and validation directories.
        """
        image_paths = self.get_image_paths()
        widths = []
        heights = []
        
        with ThreadPoolExecutor() as executor:
            future_to_image = {executor.submit(self.process_image, img_path): img_path for img_path in image_paths}
            
            for future in as_completed(future_to_image):
                width, height = future.result()
                widths.append(width)
                heights.append(height)
        
        avg_width = int(np.mean(widths))
        avg_height = int(np.mean(heights))
        
        return avg_width, avg_height
    
    def process_image(self, image_path):
        """
        Helper function to open an image and return its dimensions.
        """
        with Image.open(image_path) as img:
            return img.size
    
    def get_configs(self) -> tuple:
        """
        Get configuration from yaml.
        :return:
        """
        
        print("DEBUG: ", self.FILE_NAME, 'get_configs', 'Getting config from yaml.')
            
        if all(key in self.conf for key in ["networks", "activations", "lr"]):
            networks = self.conf["networks"]
            activations = self.conf["activations"]
            lr = float(self.conf["lr"])
            epochs = int(self.conf["epochs"])
            width_height = self.conf["width_height"]
            num_classes = int(self.conf["num_classes"])
        else:
            networks = ['VGG16', 'ResNet50V2', 'GoogleNet']
            activations = ["relu"]
            lr = 0.001
            epochs = 5
            width_height = '256,256'
            num_classes = 3
            print("ERROR: ", self.FILE_NAME, 'get_configs', 'The processor CNNTrainingProcessor needs configuration, using default configs.',
                                self.INTERNAL_SERVER_ERROR)

        return networks, activations, lr, epochs, width_height, num_classes
    
