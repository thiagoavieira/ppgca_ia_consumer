from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import cv2
import numpy as np
import torch
import time
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
# Display
import tensorflow.keras as K
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore
from tensorflow.keras.models import Model

from skimage.transform import resize
from sklearn.metrics import f1_score
from IPython.display import Image, display # type: ignore

from ..processor.abstract_processor import AbstractProcessor

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


def save_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)


def VizGradCAM(model, image, interpolant=0.5, plot_results=True):
        """VizGradCAM - Displays GradCAM based on Keras / TensorFlow models
        using the gradients from the last convolutional layer. This function
        should work with all Keras Application listed here:
        https://keras.io/api/applications/

        Parameters:
        model (keras.model): Compiled Model with Weights Loaded
        image: Image to Perform Inference On
        plot_results (boolean): True - Function Plots using PLT
                                False - Returns Heatmap Array

        Returns:
        Heatmap Array?
        """
        # Sanity Check
        assert (
            interpolant > 0 and interpolant < 1
        ), "Heatmap Interpolation Must Be Between 0 - 1"

        last_conv_layer = next(
            x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D)
        )
        target_layer = model.get_layer(last_conv_layer.name)

        original_img = image
        img = np.expand_dims(original_img, axis=0)
        prediction = model.predict(img)

        # Obtain Prediction Index
        prediction_idx = np.argmax(prediction)

        # Compute Gradient of Top Predicted Class
        with tf.GradientTape() as tape:
            gradient_model = Model([model.inputs], [target_layer.output, model.output])
            conv2d_out, prediction = gradient_model(img)
            # Obtain the Prediction Loss
            loss = prediction[:, prediction_idx]

        # Gradient() computes the gradient using operations recorded
        # in context of this tape
        gradients = tape.gradient(loss, conv2d_out)

        # Obtain the Output from Shape [1 x H x W x CHANNEL] -> [H x W x CHANNEL]
        output = conv2d_out[0]

        # Obtain Depthwise Mean
        weights = tf.reduce_mean(gradients[0], axis=(0, 1))

        # Create a 7x7 Map for Aggregation
        activation_map = np.zeros(output.shape[0:2], dtype=np.float32)

        # Multiply Weights with Every Layer
        for idx, weight in enumerate(weights):
            activation_map += weight * output[:, :, idx]

        # Resize to Size of Image
        activation_map = cv2.resize(
            activation_map.numpy(), (original_img.shape[1], original_img.shape[0])
        )

        # Ensure No Negative Numbers
        activation_map = np.maximum(activation_map, 0)

        # Convert Class Activation Map to 0 - 255
        activation_map = (activation_map - activation_map.min()) / (
            activation_map.max() - activation_map.min()
        )
        activation_map = np.uint8(255 * activation_map)

        # Convert to Heatmap
        heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

        # Superimpose Heatmap on Image Data
        original_img = np.uint8(
            (original_img - original_img.min())
            / (original_img.max() - original_img.min())
            * 255
        )

        cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Enlarge Plot
        plt.rcParams["figure.dpi"] = 100

        if plot_results == True:
            plt.imshow(
                np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant))
            )
        else:
            return cvt_heatmap

class CNNTrainingProcessor(AbstractProcessor):
    """
    Processor to organize directories to the training process.
    """
    
    FILE_NAME = 'cnn_training_processor.py'

    def __init__(self, conf):
        super(CNNTrainingProcessor, self).__init__(conf)
        self.networks, self.activations, self.lr, self.epochs, self.width_height, self.num_classes, self.fine_tune = self.get_configs()
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

        if not self.fine_tune:
            self.width = 224
            self.height = 224

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

        if self.fine_tune:
            base_model = tf.keras.applications.VGG16(
                input_shape=(self.width, self.height, 3), 
                include_top=False,  # Remove a parte totalmente conectada
                weights='imagenet'  # Mantém os pesos da ImageNet
            )
            base_model.trainable = True

            tuning_layer_name = 'block4_pool'  # Define a camada a partir de onde será feito o fine-tuning
            tuning_layer = base_model.get_layer(tuning_layer_name)
            tuning_index = base_model.layers.index(tuning_layer)

            # Congela as camadas até o tuning_index
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
        else:
            base_model = tf.keras.applications.VGG16(
                input_shape=(self.width, self.height, 3), 
                include_top=True,
                weights='imagenet'
            )
            # Remover a última camada (densa) do modelo para substituir por uma com 3 classes
            base_model.layers.pop()  
            
            x = base_model.layers[-1].output
            predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x) 

            model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

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

        final_accuracy = hist.history['val_accuracy'][-1]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")
        model_folder_name = f"{final_accuracy:.2f}_{timestamp}"
        model_directory = os.path.join(self.current_directory, '..', 'data', 'results', 'VGG16', model_folder_name)
        os.makedirs(model_directory, exist_ok=True)

        # Salvando o modelo dentro da nova pasta criada
        model_filename = f"{model_folder_name}.h5"
        model_filepath = os.path.join(model_directory, model_filename)
        model.save(model_filepath)
        print("INFO: ", self.FILE_NAME, '_process_vgg16', f"Modelo salvo em: {model_filepath}")

        y_true = testdata.classes  # Os verdadeiros do conjunto de validação
        y_score = model.predict(testdata)  # As probabilidades preditas pelo modelo
        y_pred = np.argmax(y_score, axis=1)  # As classes preditas com base nas probabilidades
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Atualizando input_data com os resultados do treinamento
        input_data['VGG16_output'] = hist.history
        input_data['VGG16_model'] = model_folder_name  # Nome do modelo/pasta
        input_data['VGG16_y_true'] = y_true.tolist()
        input_data['VGG16_y_score'] = y_score.tolist()
        input_data['VGG16_y_pred'] = y_pred.tolist()
        input_data['VGG16_f1_score'] = f1

        class_dirs = os.listdir(self.validation_path)
        if not self.fine_tune:
            for index, class_name in enumerate(class_dirs):
                class_dir_path = os.path.join(self.validation_path, class_name)
                
                if os.path.isdir(class_dir_path):
                    model_directory = os.path.dirname(model_filepath)

                    # Seleciona uma imagem aleatória de dentro dessa classe
                    class_images = [img for img in os.listdir(class_dir_path) if img.endswith('.jpg')]
                    random_image_name = random.choice(class_images)
                    random_image_path = os.path.join(class_dir_path, random_image_name)

                    xai_image_filename = f"{random_image_name.split('.')[0]}_{class_name}_xai.jpg"
                    heatmap_image_filename = f"{random_image_name.split('.')[0]}_{class_name}_heatmap.jpg"
                    original_image_filename = f"{random_image_name.split('.')[0]}_{class_name}_original.jpg"
                    xai_image_path = os.path.join(model_directory, xai_image_filename)
                    heatmap_image_path = os.path.join(model_directory, heatmap_image_filename)
                    original_image_path = os.path.join(model_directory, original_image_filename)


                    # Carrega a imagem e aplicar GradCAM
                    img = load_img(random_image_path, target_size=(224, 224))
                    img_array = img_to_array(img)
                    cv2.imwrite(original_image_path, np.uint8(img_array))
                    img = np.expand_dims(img_array, axis=0)
                    img = self.PreprocessVGG16Image(img)

                    preds = model.predict(img)
                    pred_class_idx = np.argmax(preds[0])

                    if not pred_class_idx:
                        pred_class_idx = index

                    predicted_class = class_dirs[pred_class_idx]
                    predicted_probability = preds[0][pred_class_idx]
                    print(f"Predicted class: {predicted_class} with probability: {predicted_probability:.4f}")

                    # Encontra a última camada convolucional
                    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D))
                    heatmap = make_gradcam_heatmap(img, model, last_conv_layer.name, pred_index=pred_class_idx)

                    save_gradcam(random_image_path, heatmap, xai_image_path)

                    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0])) 
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    cv2.imwrite(heatmap_image_path, heatmap_colored)

                    input_data[f'VGG16_{class_name}_gradcam_image'] = xai_image_path
                    input_data[f'VGG16_{class_name}_original_image'] = original_image_path
                    input_data[f'VGG16_{class_name}_heatmap_image'] = heatmap_image_path


        return input_data

    def _process_resnet50v2(self, input_data: dict, device: str) -> dict:
        print("INFO: ", self.FILE_NAME, '_process_resnet50v2',f"Running ResNet50V2 on {device}")

        if not self.fine_tune:
            self.width = 224
            self.height = 224

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

        if self.fine_tune:
            base_model = tf.keras.applications.ResNet50V2(
                input_shape=(self.width, self.height, 3), 
                include_top=False, 
                weights='imagenet'
            )
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

        else:
            base_model = tf.keras.applications.ResNet50V2(
                input_shape=(self.width, self.height, 3),
                include_top=True,
                weights='imagenet'
            )
            # Remover a última camada (densa) do modelo para substituir por uma com 3 classes
            base_model.layers.pop()  
            
            x = base_model.layers[-1].output
            predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x) 

            model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        
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

        final_accuracy = hist.history['val_accuracy'][-1]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")
        model_folder_name = f"{final_accuracy:.2f}_{timestamp}"
        model_directory = os.path.join(self.current_directory, '..', 'data', 'results', 'ResNet50v2', model_folder_name)
        os.makedirs(model_directory, exist_ok=True)

        model_filename = f"{model_folder_name}.h5"
        model_filepath = os.path.join(model_directory, model_filename)
        model.save(model_filepath)
        print("INFO: ", self.FILE_NAME, '_process_resnet50v2', f"Modelo salvo em: {model_filepath}")

        y_true = testdata.classes  # Os verdadeiros do conjunto de validação
        y_score = model.predict(testdata)  # As probabilidades preditas pelo modelo
        y_pred = np.argmax(y_score, axis=1)  # As classes preditas com base nas probabilidades
        f1 = f1_score(y_true, y_pred, average='weighted')

        input_data['ResNet50v2_output'] = hist.history
        input_data['ResNet50v2_model'] = model_folder_name  # Nome do modelo/pasta
        input_data['ResNet50v2_y_true'] = y_true.tolist()
        input_data['ResNet50v2_y_score'] = y_score.tolist()
        input_data['ResNet50v2_y_pred'] = y_pred.tolist()
        input_data['ResNet50v2_f1_score'] = f1

        class_dirs = os.listdir(self.validation_path)
        if not self.fine_tune:
            for index, class_name in enumerate(class_dirs):
                class_dir_path = os.path.join(self.validation_path, class_name)
                
                if os.path.isdir(class_dir_path):
                    model_directory = os.path.dirname(model_filepath)

                    # Seleciona uma imagem aleatória de dentro dessa classe
                    class_images = [img for img in os.listdir(class_dir_path) if img.endswith('.jpg')]
                    random_image_name = random.choice(class_images)
                    random_image_path = os.path.join(class_dir_path, random_image_name)

                    xai_image_filename = f"{random_image_name.split('.')[0]}_{class_name}_xai.jpg"
                    heatmap_image_filename = f"{random_image_name.split('.')[0]}_{class_name}_heatmap.jpg"
                    original_image_filename = f"{random_image_name.split('.')[0]}_{class_name}_original.jpg"
                    xai_image_path = os.path.join(model_directory, xai_image_filename)
                    heatmap_image_path = os.path.join(model_directory, heatmap_image_filename)
                    original_image_path = os.path.join(model_directory, original_image_filename)

                    # Carrega a imagem e aplicar GradCAM
                    img = load_img(random_image_path, target_size=(224, 224))
                    img_array = img_to_array(img)
                    img = np.expand_dims(img_array, axis=0)
                    img = self.PreprocessVGG16Image(img)

                    preds = model.predict(img)
                    pred_class_idx = np.argmax(preds[0])

                    if not pred_class_idx:
                        pred_class_idx = index

                    predicted_class = class_dirs[pred_class_idx]
                    predicted_probability = preds[0][pred_class_idx]
                    print(f"Predicted class: {predicted_class} with probability: {predicted_probability:.4f}")

                    # Encontra a última camada convolucional
                    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D))
                    heatmap = make_gradcam_heatmap(img, model, last_conv_layer.name, pred_index=pred_class_idx)

                    save_gradcam(random_image_path, heatmap, xai_image_path)
                    img_rgb = np.uint8(img_array)  # Converter para uint8 (0-255)
                    cv2.imwrite(original_image_path, img_rgb)

                    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0])) 
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    cv2.imwrite(heatmap_image_path, heatmap_colored)

                    input_data[f'ResNet50v2_{class_name}_gradcam_image'] = xai_image_path
                    input_data[f'ResNet50v2_{class_name}_original_image'] = original_image_path
                    input_data[f'ResNet50v2_{class_name}_heatmap_image'] = heatmap_image_path
        return input_data

    def _process_googlenet(self, input_data: dict, device: str) -> dict:
        print("INFO: ", self.FILE_NAME, '_process_googlenet',f"Running GoogleNet on {device}")

        if not self.fine_tune:
            self.width = 299
            self.height = 299

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

        if self.fine_tune:
            base_model = tf.keras.applications.InceptionV3(
                input_shape=(self.width, self.height, 3), 
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = True

            tuning_layer_name = 'mixed7'
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

        else:
            base_model = tf.keras.applications.InceptionV3(
                input_shape=(self.width, self.height, 3),
                include_top=True, 
                weights='imagenet'
            )
            
            # Remover a última camada (densa) do modelo para substituir por uma com 3 classes
            base_model.layers.pop()  
            
            x = base_model.layers[-1].output
            predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x) 

            model = tf.keras.Model(inputs=base_model.input, outputs=predictions)


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
        
        final_accuracy = hist.history['val_accuracy'][-1]
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M")
        model_folder_name = f"{final_accuracy:.2f}_{timestamp}"
        model_directory = os.path.join(self.current_directory, '..', 'data', 'results', 'GoogLeNet', model_folder_name)
        os.makedirs(model_directory, exist_ok=True)

        # Salvando o modelo dentro da nova pasta criada
        model_filename = f"{model_folder_name}.h5"
        model_filepath = os.path.join(model_directory, model_filename)
        model.save(model_filepath)
        print("INFO: ", self.FILE_NAME, '_process_googlenet', f"Modelo salvo em: {model_filepath}")

        y_true = testdata.classes  # Os verdadeiros do conjunto de validação
        y_score = model.predict(testdata)  # As probabilidades preditas pelo modelo
        y_pred = np.argmax(y_score, axis=1)  # As classes preditas com base nas probabilidades
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Atualizando input_data com os resultados do treinamento
        input_data['GoogLeNet_output'] = hist.history
        input_data['GoogLeNet_model'] = model_folder_name  # Nome do modelo/pasta
        input_data['GoogLeNet_y_true'] = y_true.tolist()
        input_data['GoogLeNet_y_score'] = y_score.tolist()
        input_data['GoogLeNet_y_pred'] = y_pred.tolist()
        input_data['GoogLeNet_f1_score'] = f1

        class_dirs = os.listdir(self.validation_path)
        if not self.fine_tune:
            for index, class_name in enumerate(class_dirs):
                class_dir_path = os.path.join(self.validation_path, class_name)
                
                if os.path.isdir(class_dir_path):
                    model_directory = os.path.dirname(model_filepath)

                    # Seleciona uma imagem aleatória de dentro dessa classe
                    class_images = [img for img in os.listdir(class_dir_path) if img.endswith('.jpg')]
                    random_image_name = random.choice(class_images)
                    random_image_path = os.path.join(class_dir_path, random_image_name)

                    xai_image_filename = f"{random_image_name.split('.')[0]}_{class_name}_xai.jpg"
                    heatmap_image_filename = f"{random_image_name.split('.')[0]}_{class_name}_heatmap.jpg"
                    original_image_filename = f"{random_image_name.split('.')[0]}_{class_name}_original.jpg"
                    xai_image_path = os.path.join(model_directory, xai_image_filename)
                    heatmap_image_path = os.path.join(model_directory, heatmap_image_filename)
                    original_image_path = os.path.join(model_directory, original_image_filename)

                    # Carrega a imagem e aplicar GradCAM
                    img = load_img(random_image_path, target_size=(299, 299))
                    img_array = img_to_array(img)
                    img = np.expand_dims(img_array, axis=0)
                    img = self.PreprocessVGG16Image(img)

                    preds = model.predict(img)
                    pred_class_idx = np.argmax(preds[0])

                    if not pred_class_idx:
                        pred_class_idx = index

                    predicted_class = class_dirs[pred_class_idx]
                    predicted_probability = preds[0][pred_class_idx]
                    print(f"Predicted class: {predicted_class} with probability: {predicted_probability:.4f}")

                    # Encontra a última camada convolucional
                    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D))
                    heatmap = make_gradcam_heatmap(img, model, last_conv_layer.name, pred_index=pred_class_idx)

                    save_gradcam(random_image_path, heatmap, xai_image_path)
                    img_rgb = np.uint8(img_array)  # Converter para uint8 (0-255)
                    cv2.imwrite(original_image_path, img_rgb)

                    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0])) 
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    cv2.imwrite(heatmap_image_path, heatmap_colored)

                    input_data[f'GoogLeNet_{class_name}_gradcam_image'] = xai_image_path
                    input_data[f'GoogLeNet_{class_name}_original_image'] = original_image_path
                    input_data[f'GoogLeNet_{class_name}_heatmap_image'] = heatmap_image_path

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
        
    def get_img_array(self, img_path, size):
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array


    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def save_and_display_gradcam(self, img, heatmap, cam_path, alpha=0.8):
        img = np.uint8(255 * img)
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        superimposed_img.save(cam_path)
        display(Image(cam_path))
        return superimposed_img

    def PreprocessVGG16Image(self, im):
        im = tf.keras.applications.vgg16.preprocess_input(im)
        return im
    
    def PreprocessInceptionImage(self, im):
        im = tf.keras.applications.inception_v3.preprocess_input(im)
        return im

    def PreprocessResNet50v2Image(self, im):
        im = tf.keras.applications.resnet50v2.preprocess_input(im)
        return im
    
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
            fine_tune = int(self.conf["fine_tune"])
        else:
            networks = ['VGG16', 'ResNet50V2', 'GoogleNet']
            activations = ["relu"]
            lr = 0.001
            epochs = 5
            width_height = '256,256'
            num_classes = 3
            fine_tune = True
            print("ERROR: ", self.FILE_NAME, 'get_configs', 'The processor CNNTrainingProcessor needs configuration, using default configs.',
                                self.INTERNAL_SERVER_ERROR)

        return networks, activations, lr, epochs, width_height, num_classes, fine_tune
    
