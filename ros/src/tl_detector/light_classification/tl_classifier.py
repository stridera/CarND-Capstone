from abc import ABCMeta, abstractmethod


class TLClassifier:
    """
    Abstract class for traffic light classification from RGB images
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_classification(self, image):
        raise NotImplementedError()

