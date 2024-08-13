import abc


class AbstractProcessor(metaclass=abc.ABCMeta):
    """
    Default class to process pipeline.
    """
    def __init__(self, conf):
        self.conf = conf
        self.INTERNAL_SERVER_ERROR = 500

    def execute(self, input) -> dict:
        """
        Execute the abstract methods to process something (input data).
        The sequence is pre_process and after process.
        :param input:
        :return:
        """
        input_data = self.pre_process(input)
        return self.process(input_data)

    @abc.abstractmethod
    def pre_process(self, input_data) -> dict:
        """
        Pre process input data
        :param input_data:
        :return:
        """
        pass

    @abc.abstractmethod
    def process(self, input_data) -> dict:
        """
        Process input data
        :param input_data:
        :return:
        """
        pass
