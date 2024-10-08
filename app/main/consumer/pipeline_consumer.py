import os
import sys
from datetime import datetime
from omegaconf import OmegaConf

from ..util.enum_constants import Constants

class PipelineConsumer:
    """
    Consumer to initiate the entire pipeline.
    """
    FILE_NAME = 'pipeline_consumer.py'

    def __init__(self, processors, config_filename):
        final_processor_names = ["CNNTrainingProcessor", "EvaluationMetricsProcessor"]
        self.processors = [p for p in processors if p.__class__.__name__ not in final_processor_names]
        self.final_processors = [p for p in processors if p.__class__.__name__ in final_processor_names]
        self.thread_mode = False
        self.last_log_time = datetime.now()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'configuration', config_filename)
        self.config_filename = config_filename
        self.config = OmegaConf.load(config_path)

    def main(self):
        if self.thread_mode:
            print("DEBUG: ", self.FILE_NAME, 'main', "Running with threads!")
            self._run_with_threads()
        else:
            print("DEBUG: ", self.FILE_NAME, 'main', "Running without threads!")
            self._run_without_threads()

    def _run_without_threads(self):
        teeth_dict = self.config.consumer.teeth_dict
        
        for description, teeth_types in teeth_dict.items():
            single_dict = {
                description: teeth_types
                }
            print("DEBUG: ", self.FILE_NAME, '_run_without_threads', f"Dict to be sent to pipeline: {single_dict}")
            self.pipeline(single_dict, "processors")
        
        self.pipeline({"run": self.config_filename}, "final_processors")

    def _run_with_threads(self):
        """
            TODO
        """

    def pipeline(self, teeth_type_dict, processors_type: str):
        """
        Execute the pipeline of the processor. The input of the next processor is always the exit of before processor.
        :param teeth_type_dict:
        :return:
        """
        try:
            # print(self.FILE_NAME, "Thread ID", threading.get_ident())  # In case you want to see the TID
            self.processing_pipeline(teeth_type_dict, processors_type)
        except Exception as err:
            if hasattr(err, 'message'):
                print("ERROR: ", self.FILE_NAME, 'main', "Unknown Error pipeline function 1: " + err.message)
            else:
                print("ERROR: ", self.FILE_NAME, 'main', "Unknown Error pipeline function 2: " + str(err))

            exc_type, exc_obj, tb = sys.exc_info()
            print("ERROR: ", self.FILE_NAME, 'main', "Error type: '{}'. Description: '{}'. Line: '{}'".format(str(type(err).__name__), str(exc_obj), str(tb.tb_lineno)))
        finally:
            if self.thread_mode is True:
                print("TODO")
                # Remove teeth_type from sequential queue when finished processing/process


    def processing_pipeline(self, teeth_type_dict: dict, processors_type: str):
        """
        Sending all lists to available processors in order
        :param teeth_type_dict: is the input of each processor
        """
        try:
            processors = getattr(self, processors_type, [])
            if len(processors) > 0:
                input_data = None
                for processor in processors:
                    start = datetime.now()
                    if self.thread_mode is True:
                        # TODO
                        print("TODO")
                    else:
                        if input_data is None:
                            input_data = processor.execute(teeth_type_dict)
                        else:
                            if input_data is Constants.ignore_message:
                                break
                            input_data = processor.execute(input_data)

                    processor_time = datetime.now() - start
                    print("DEBUG: ", self.FILE_NAME, 'pipeline', '--> processor_time: {} - {}'.format(
                        str(processor_time.total_seconds()), str(processor.__class__.__name__)))

                self.ping_log('Alive!', interval=60)
        except Exception as err:
            if hasattr(err, 'message'):
                print("ERROR: ", self.FILE_NAME, 'main', 'Unknown Error 1: ' + str(err))
            else:
                exc_type, exc_obj, tb = sys.exc_info()
                print("ERROR: ", self.FILE_NAME, 'main', "Error type: '{}'. Processor: '{}'. Line: '{}'".format(
                    str(type(err).__name__), str(processor.__class__.__name__), str(tb.tb_lineno)))
                if str(exc_obj) == "''":
                    print("WARNING: ", self.FILE_NAME, 'main', "Some value is missing! Please check out [Input_data]")
                print("WARNING: ", self.FILE_NAME, 'main', "Error description: ", str(err))



    def ping_log(self, message, interval=60):
        """
        Ping an info log respecting the interval time
        :param message: is the log message
        :param interval: is the interval in seconds
        """
        current = datetime.now()
        diff_time = current - self.last_log_time
        if diff_time.total_seconds() >= interval:
            print("INFO: ", self.FILE_NAME, 'ping_log', message)
            self.last_log_time = current
