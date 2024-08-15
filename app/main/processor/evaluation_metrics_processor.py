import os
import json
import random
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from ..processor.abstract_processor import AbstractProcessor


class EvaluationMetricsProcessor(AbstractProcessor):
    """
    Processor to organize directories to the training process.
    """
    
    FILE_NAME = 'evaluation_metrics_processor.py'

    def __init__(self, conf):
        super(EvaluationMetricsProcessor, self).__init__(conf)

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

        current_directory = Path(__file__).resolve().parent
        results_directory = current_directory.parent / "data" / "results"
        
        for key, value in input_data.items():
            if '_output' in key:
                model_name = key.replace('_output', '')
                model_results = value
                model_dir_name = input_data.get(f'{model_name}_model', 'default_model_name')
                
                # Criar o diretório para salvar os gráficos
                model_dir = results_directory / model_name / model_dir_name
                os.makedirs(model_dir, exist_ok=True)
                
                # Identificar as chaves específicas para GoogLeNet
                acc_key = next((k for k in model_results if 'accuracy' in k and 'val' not in k), 'accuracy')
                val_acc_key = next((k for k in model_results if 'val' in k and 'accuracy' in k), 'val_accuracy')
                loss_key = 'loss'
                val_loss_key = 'val_loss'
                
                # Plotar a acurácia
                plt.figure()
                plt.plot(model_results[acc_key], label='Training Accuracy')
                plt.plot(model_results[val_acc_key], label='Validation Accuracy')
                plt.title(f'{model_name} Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.savefig(model_dir / f'{model_name}_accuracy_plot.png')
                plt.close()

                # Plotar a perda
                plt.figure()
                plt.plot(model_results[loss_key], label='Training Loss')
                plt.plot(model_results[val_loss_key], label='Validation Loss')
                plt.title(f'{model_name} Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(model_dir / f'{model_name}_loss_plot.png')
                plt.close()

                print(f'Plots saved in {model_dir}')
        
        return input_data
