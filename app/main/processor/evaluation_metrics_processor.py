import os
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from pathlib import Path

import yaml
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
                
                # chaves
                acc_key = 'accuracy'
                val_acc_key = 'val_accuracy'
                loss_key = 'loss'
                val_loss_key = 'val_loss'
                
                # Accuraccy plot
                plt.figure()
                plt.plot(model_results[acc_key], label='Training Accuracy')
                plt.plot(model_results[val_acc_key], label='Validation Accuracy')
                max_train_acc = max(model_results[acc_key])
                max_val_acc = max(model_results[val_acc_key])
                plt.text(len(model_results[acc_key]) - 1, max_train_acc, f'{max_train_acc:.2f}', color='blue', fontsize=12)
                plt.text(len(model_results[val_acc_key]) - 1, max_val_acc, f'{max_val_acc:.2f}', color='orange', fontsize=12)
                plt.title(f'{model_name} Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.savefig(model_dir / f'{model_name}_accuracy_plot.png')
                plt.close()

                # Loss plot
                plt.figure()
                plt.plot(model_results[loss_key], label='Training Loss')
                plt.plot(model_results[val_loss_key], label='Validation Loss')
                min_train_loss = min(model_results[loss_key])
                min_val_loss = min(model_results[val_loss_key])
                plt.text(len(model_results[loss_key]) - 1, min_train_loss, f'{min_train_loss:.2f}', color='blue', fontsize=12)
                plt.text(len(model_results[val_loss_key]) - 1, min_val_loss, f'{min_val_loss:.2f}', color='orange', fontsize=12)
                plt.title(f'{model_name} Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(model_dir / f'{model_name}_loss_plot.png')
                plt.close()

                print("INFO: ", self.FILE_NAME, 'process', f'Plots saved in {model_dir}')
                
                # Calcular a curva ROC e AUC
                y_true = input_data[f'{model_name}_y_true']
                y_score = input_data[f'{model_name}_y_score']
                y_pred = input_data[f'{model_name}_y_pred']

                classes = np.unique(y_true)
                y_true_bin = label_binarize(y_true, classes=classes)

                for i, class_label in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], np.array(y_score)[:, i])
                    roc_auc = auc(fpr, tpr)

                    plt.figure()
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{model_name} ROC Curve for class {class_label}')
                    plt.legend(loc="lower right")
                    plt.savefig(model_dir / f'{model_name}_roc_curve_class_{class_label}.png')
                    plt.close()

                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], np.array(y_score)[:, i])
                    f1 = f1_score(y_true, y_pred, average='weighted')

                    plt.figure()
                    plt.plot(recall, precision, lw=2, color='blue', label=f'{model_name} PR curve')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'{model_name} Precision-Recall Curve for class {class_label}')
                    plt.text(0.6, 0.2, f'F1 Score: {f1:.2f}', fontsize=12, color='red',
                            bbox=dict(facecolor='white', alpha=0.8))  # Caixa de fundo branco para o texto do f1score
                    plt.legend(loc="lower left")
                    plt.savefig(model_dir / f'{model_name}_precision_recall_curve_class_{class_label}.png')
                    plt.close()

                print("INFO: ", self.FILE_NAME, 'process', f'Plots saved in {model_dir}')

                run_name = input_data.get("run", "default_config.yaml")
                yaml_file_path = model_dir / run_name

                config_dir = Path(__file__).resolve().parent.parent / 'configuration'
                config_file_path = config_dir / run_name

                try:
                    with open(config_file_path, 'r') as config_file:
                        yaml_content = yaml.safe_load(config_file)
                except FileNotFoundError:
                    yaml_content = {}
                    print(f"WARNING: {self.FILE_NAME}", 'process', f"File {run_name} not found, saving an empty YAML file.")

                with open(yaml_file_path, 'w') as yaml_file:
                    yaml.dump(yaml_content, yaml_file)
                print(f"INFO: {self.FILE_NAME}", 'process', f"Created {run_name}.yaml in {model_dir}")

                input_data_path = model_dir / 'input_data.txt'
                with open(input_data_path, 'w') as input_data_file:
                    json.dump(input_data, input_data_file, indent=4)
                print(f"INFO: {self.FILE_NAME}", 'process', f"Saved input_data in {input_data_path}")
                    
        return input_data
