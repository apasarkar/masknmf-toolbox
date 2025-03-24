from masknmf.arrays.array_interfaces import LazyFrameLoader
import torch


class RegistrationArray(LazyFrameLoader):

    def __init__(self, template, method, **params):