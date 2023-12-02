import tensorflow as tf
from Network import Network
from tqdm import tqdm
import numpy as np

class HeatModel:
    # 1D Heat equation PINN-solver. It contains its own nn.
    def __init__(self):

        self.model = Network([2,500,500,500,1]).build()
    
        self.mu = 0.001

    def init_condition(self, data):
        n = data.shape[0]
        return tf.reshape(tf.math.sin(data[...,1] * np.pi), shape=(n, 1))
    
    ##################### LOSS METHODS ###########################

    def pde_loss(self, data):

        # Error given by the PDE over data.
        
        n = data.shape[0]
        r = tf.reshape(tf.math.exp(-data[...,0]) * (tf.math.sin(data[...,1] * np.pi) - (np.pi**2) * tf.math.sin(data[...,1] * np.pi)), shape=(n,1))
        with tf.GradientTape() as tape2:
            tape2.watch(data)
            with tf.GradientTape() as tape1:
                tape1.watch(data)
                u = self.model(data)
            u_z = tape1.gradient(u, data)
        u_zz = tape2.gradient(u_z, data)
        u_t = tf.reshape(u_z[...,0], shape=(n,1))
        u_xx = tf.reshape(u_zz[...,1], shape=(n,1))
        return tf.reduce_mean(tf.math.square(u_t - u_xx + r))

    def boundary_loss(self, data_init, data_left, data_right):
        l1 = tf.reduce_mean(tf.math.square(self.model(data_left)))
        l2 = tf.reduce_mean(tf.math.square(self.model(data_right)))
        l3 = tf.reduce_mean(tf.math.square(self.model(data_init) - self.init_condition(data_init)))
        return l1 + l2 + l3
    
    def total_loss(self, data_int, data_init, data_left, data_right):
    
        #self.boundary_loss(data_init, data_left, data_right) + 
        return (1 - self.mu) * self.boundary_loss(data_init, data_left, data_right) + self.mu * self.pde_loss(data_int)
    ##################### TRAINING METHODS ###########################
    
    @tf.function
    def gradients(self, data_int, data_init, data_left, data_right):
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            target = self.total_loss(data_int, data_init, data_left, data_right)
        return target, tape.gradient(target, self.model.trainable_variables)

    def fit_SGD(self, data_int, data_init, data_left, data_right):
        # Random mini-batch
        n = data_int.shape[0]
        l = []
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        b = 100
        for _ in tqdm(tf.range(5000)):
            i = np.random.randint(n - b)
            target, gradients = self.gradients(data_int[i:i+b], data_init[i:i+b], data_left[i:i+b], data_right[i:i+b])
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            l.append(target)
            if target < 1e-4:
                break
        return l
    
    def fit(self, data_int, data_init, data_left, data_right):
        l = []
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        for epoch in tqdm(tf.range(2500)):
            target, gradients = self.gradients(data_int, data_init, data_left, data_right)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            l.append(target)
            if target.numpy() < 1e-2:
                break
        return l
