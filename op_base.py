import tensorflow as tf
import os

class op_base():
    def __init__(self,args):
        self.__dict__ = args.__dict__
