import tensorflow as tf
import sys
sys.path.append('D:/Math PhD/PINNs code and papers/PINNs')   
from Network import Network
from tqdm import tqdm
import numpy as np

class BurgersModel:
    # 1D Heat equation PINN-solver. It contains its own nn.
    def __init__(self, architecture, mu, learning_rate, opti_iter, mini_batch):

        self.model = Network(architecture).build()
        self.mu = mu
        self.learning_rate = learning_rate
        self.opti_iter = opti_iter
        self.mini_batch = mini_batch
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def init_condition(self, data):
        n = data.shape[0]
        return tf.reshape(tf.math.sin(data[...,1] * (-np.pi)), shape=(n, 1))
    
    ##################### LOSS METHODS ###########################

    def pde_loss(self, data):

        # Error given by the PDE over data.

        n = data.shape[0]
        with tf.GradientTape() as tape2:
            tape2.watch(data)
            with tf.GradientTape() as tape1:
                tape1.watch(data)
                u = self.model(data)
            u_z = tape1.gradient(u, data)
        u_zz = tape2.gradient(u_z, data)
        u_t = tf.reshape(u_z[...,0], shape=(n,1))
        u_x = tf.reshape(u_z[...,1], shape=(n,1))
        u_xx = tf.reshape(u_zz[...,1], shape=(n,1))
        return tf.reduce_mean(tf.math.square(u_t + u*u_x - (0.01/np.pi)*u_xx))

    def boundary_loss(self, data_init, data_left, data_right):
        l1 = tf.reduce_mean(tf.math.square(self.model(data_left)))
        l2 = tf.reduce_mean(tf.math.square(self.model(data_right)))
        l3 = tf.reduce_mean(tf.math.square(self.model(data_init) - self.init_condition(data_init)))
        return l1 + l2 + l3
    
    def total_loss(self, data_int, data_init, data_left, data_right):
    
        #self.boundary_loss(data_init, data_left, data_right) + 
        return (1-self.mu) * self.boundary_loss(data_init, data_left, data_right) + self.mu*self.pde_loss(data_int)
    ##################### TRAINING METHODS ###########################
    
    def gradients(self, data_int, data_init, data_left, data_right):
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            target = self.total_loss(data_int, data_init, data_left, data_right)
        return target, tape.gradient(target, self.model.trainable_variables)

    @tf.function
    def fit_Adam(self, data_int, data_init, data_left, data_right):
        # Performs stochastic descent
        n = data_int.shape[0]
        l = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        b = self.mini_batch
        for _ in tf.range(self.opti_iter):
            i = np.random.randint(n - b)
            target, gradients = self.gradients(data_int[i:i+b], data_init[i:i+b], data_left[i:i+b], data_right[i:i+b])
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            l = l.write(_, target)
        return l.stack()
    
    def fit(self, data_int, data_init, data_left, data_right):
        l = []
        for epoch in tqdm(tf.range(self.learning_rate)):
            target, gradients = self.gradients(data_int, data_init, data_left, data_right)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            l.append(target)
        return l

