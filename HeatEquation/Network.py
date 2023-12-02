import keras
from keras import layers

class Network:
    def __init__(self, layers_list: list) -> None:
        self.layers_list = layers_list

    def build(self):
        input_dim = self.layers_list.pop(0)
        ouput_dim = self.layers_list.pop()
        
        array = [layers.Input(shape=(input_dim,))]
        for units in self.layers_list:
            array.append(layers.Dense(units, activation="tanh", kernel_initializer="he_normal", bias_initializer="zeros"))
        array.append(layers.Dense(ouput_dim, kernel_initializer="he_normal", use_bias=False))
        
        nn = keras.Sequential(array)

        return nn