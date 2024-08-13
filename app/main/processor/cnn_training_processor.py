import json
import torch
import time
import sklearn
import keras, os
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
# For VGG16
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
# For ResNet50V2
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from ..processor.abstract_processor import AbstractProcessor


class CNNTrainingProcessor(AbstractProcessor):
    """
    Processor to organize directories to the training process.
    """
    
    FILE_NAME = 'cnn_training_processor.py'

    def __init__(self, conf):
        super(CNNTrainingProcessor, self).__init__(conf)
        self.networks, self.activations, self.lr = self.get_configs()
        self.gpu_count = torch.cuda.device_count()
        self.device_queue = list(range(self.gpu_count)) if self.gpu_count > 0 else ['cpu']

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
        :param input_data: Dados de entrada.
        :param network: Nome da rede.
        :param device: Dispositivo (GPU ou CPU) onde o processamento ocorrerá.
        :return: Dados processados.
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

        current_directory = os.path.dirname(os.path.abspath(__file__))
        training_path = os.path.join(current_directory, '..', 'data', 'cropped_images', 'training')
        validation_path = os.path.join(current_directory, '..', 'data', 'cropped_images', 'training')

        trdata = ImageDataGenerator()
        traindata = trdata.flow_from_directory(
            directory=training_path,
            target_size=(224,224),
            batch_size=32,
            class_mode='categorical'
        )
        tsdata = ImageDataGenerator()
        testdata = tsdata.flow_from_directory(
            directory=validation_path,
            target_size=(224,224),
            batch_size=32,
            class_mode='categorical'
        )

        model = Sequential()
        model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=2, activation="softmax"))

        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        model.summary()

        checkpoint = ModelCheckpoint("vgg16_1.keras", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', save_freq='epoch')
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='max')
        hist = model.fit(steps_per_epoch=len(traindata), x=traindata, validation_data=testdata, validation_steps=len(testdata), epochs=1, callbacks=[checkpoint, early])

        results_directory = os.path.join(current_directory, '..', 'data', 'results', 'VGG16')
        os.makedirs(results_directory, exist_ok=True)
        final_accuracy = hist.history['accuracy'][-1]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")

        model_filename = f"{final_accuracy:.2f}_{timestamp}.h5"
        model_filepath = os.path.join(results_directory, model_filename)
        
        model.save(model_filepath)
        print(f"Modelo salvo em: {model_filepath}")

        input_data['VGG16_output'] = hist.history

        return input_data

    def _process_resnet50v2(self, input_data: dict, device: str) -> dict:
        print("INFO: ", self.FILE_NAME, '_process_resnet50v2',f"Running ResNet50V2 on {device}")

        current_directory = os.path.dirname(os.path.abspath(__file__))
        training_path = os.path.join(current_directory, '..', 'data', 'cropped_images', 'training')
        validation_path = os.path.join(current_directory, '..', 'data', 'cropped_images', 'training')

        trdata = ImageDataGenerator()
        traindata = trdata.flow_from_directory(
            directory=training_path,
            target_size=(256,256),
            batch_size=32,
            class_mode='categorical'
        )
        tsdata = ImageDataGenerator()
        testdata = tsdata.flow_from_directory(
            directory=validation_path,
            target_size=(256,256),
            batch_size=32,
            class_mode='categorical'
        )

        base_model = tf.keras.applications.ResNet50V2(input_shape=(256,256,3), include_top=False)

        base_model.trainable = True
        base_model.summary()

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
            data_transforms, base_model, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(2, activation='softmax')
        ])
        learning_rate = 0.001
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

        hist = model.fit(traindata, validation_data=testdata, epochs=1)

        results_directory = os.path.join(current_directory, '..', 'data', 'results', 'ResNet50v2')
        os.makedirs(results_directory, exist_ok=True)
        final_accuracy = hist.history['accuracy'][-1]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")

        model_filename = f"{final_accuracy:.2f}_{timestamp}.h5"
        model_filepath = os.path.join(results_directory, model_filename)
        
        model.save(model_filepath)
        print(f"Modelo salvo em: {model_filepath}")

        input_data['ResNet50V2_output'] = hist.history
        return input_data

    def _process_googlenet(self, input_data: dict, device: str) -> dict:
        print("INFO: ", self.FILE_NAME, '_process_googlenet',f"Running GoogleNet on {device}")
        input_data['GoogleNet_output'] = f'processed_data_googlenet_on_{device}'
        return input_data
   
    def get_configs(self) -> tuple:
        """
        Get configuration from yaml.
        :return:
        """
        
        print("DEBUG: ", self.FILE_NAME, 'get_configs', 'Getting config from yaml.')
            
        if all(key in self.conf for key in ["networks", "activations", "lr"]):
            networks = self.conf["networks"]
            activations = self.conf["activations"]
            lr = self.conf["lr"]
        else:
            networks = ['VGG16', 'ResNet50V2', 'GoogleNet']
            activations = ["relu"]
            lr = 0.001
            print("ERROR: ", self.FILE_NAME, 'get_configs', 'The processor CNNTrainingProcessor needs configuration, using default configs.',
                                self.INTERNAL_SERVER_ERROR)

        return networks, activations, lr
    
