import numpy as np
import tensorflow as tf
import tqdm

class HeatModel:
    # Heat equation PINN-solver. 
    def __init__(self, layers_size):

        self.layers = layers_size
        # List of tf.Tensor objects
        self.W = []

        # Initializer
        init = tf.initializers.GlorotUniform(seed=333)

        for i in range(len(layers_size) - 1):
            input_dim = layers_size[i]
            output_dim = layers_size[i + 1]

            w = tf.Variable(init([output_dim, input_dim], dtype='float64'), trainable=True, name=f'w{i+1}')
            b = tf.zeros([output_dim, 1], dtype='float64')
            b = tf.Variable(b, trainable=True, name=f'b{i+1}')

            self.W.append(w)
            self.W.append(b)

        # Learning rate
        self.rate = 0.001
    
    

    def evaluate(self, x):
        """Evaluates the NN at x.

        Args:
            x : Shape must match the NN input shape.

        Returns:
            tf.Tensor: u_theta(x)
        """
        a = x
        for i in range(len(self.layers) - 2):
            z = tf.add(tf.matmul(self.W[2*i], a), self.W[2*i + 1])
            a = tf.nn.tanh(z)

        a = tf.add(tf.matmul(self.W[-2], a), self.W[-1])
        return a
     
    def get_weights(self):
        return self.W
    
    def set_weights(self, new_weights):
        # Setter for the parameters.
        # Shapes must coincide.
        for i in range(len(new_weights)):
            self.W[i].assign(new_weights[i])
        self.get_weights() 

   
    ##################### LOSS METHODS ###########################
    def initial_condition_loss(self, init_data):
        return tf.square(tf.add(self.evaluate(init_data), 1)) 
        
            
    def boundary_loss(self, boundary_data, init_data):
        """Computes boundary loss at one data point.

        Args:
            boundary_data (_type_): _description_
            init_data (_type_): _description_

        Returns:
            tf.Tensor: shape ().
        """
        init_data = tf.reshape(tf.Variable(init_data, trainable=False), (3,1))

        l1 = tf.square(tf.add(self.evaluate(init_data), 1))
        l2 = tf.square

        return None
        
            
    def physics_loss(self, int_data):

        # Error given by the PDE over data_set.
        # We compute the first and second derivative w.r.t the NN variable.
        # returns tf.Tensor shape=().
        # Every derivative is reshaped to shape (1,1).
        
        z = tf.reshape(tf.Variable(int_data, trainable=False), (3,1))
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                tape1.watch(z)
                u = self.evaluate(z)
            u_z = tape1.gradient(u, z)
            tape2.watch(z)
        u_zz = tape2.gradient(u_z, z)

        u_t = tf.reshape(u_z[0], (1,1))
        u_xx = tf.reshape(u_zz[1], (1,1))
        u_yy = tf.reshape(u_zz[2], (1,1))

        del tape1, tape2

        aux = tf.add(u_xx, u_yy)
        aux = tf.subtract(u_t, aux)

        return tf.square(aux)


  

    def total_loss(self, boundary_data, init_data, int_data):
        # Loss computed from the PDE + loss from training data.
        # returns: tf.Tensor shape=()

        return tf.add(self.boundary_loss(boundary_data, init_data), self.physics_loss(int_data))

    ##################### TRAINING METHODS ###########################

    def gradients(self, boundary_data, init_data, int_data):
        # This function computes the gradient of total_loss() method
        # w.r.t trainable_variables.
        # Returns gradients evaluated at the current data point and
        # trainable variable, also returns the current value of the
        # loss function.
        # Since self.trainable_variables is constantly being updated,
        # the gradient is evaluated at different values of weights.

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            target = self.total_loss(boundary_data, init_data, int_data)
        gradients = tape.gradient(target, self.trainable_variables)
        return gradients, target

    
    def generate_data_sets(self):

        N = self.INTERIOR
        boundary_data = []
        
        # interior data set.
        T = np.random.uniform(0, 1, N)
        X = np.random.uniform(0, 1, N)
        Y = np.random.uniform(0, 1, N)

        int_data = [[[t], [x], [y]] for t,x,y in zip(T,X,Y)]

        # initial condition data set.
        X = np.random.uniform(0, 1, N)
        Y = np.random.uniform(0, 1, N)
        init_data = [[[0], [x], [y]] for t,x,y in zip(T,X,Y)]



        # boundary data        
        X = np.random.uniform(0, 1, N)
        T = np.random.uniform(0, 1, N)
        up = [[[t],[x],[1]] for t,x in zip(T,X)]
        Y = np.random.uniform(0, 1, N)
        T = np.random.uniform(0, 1, N)
        left = [[[t],[0],[y]] for t,y in zip(T,Y)]
        X = np.random.uniform(0, 1, N)
        T = np.random.uniform(0, 1, N)
        down = [[[t],[x],[0]] for t,x in zip(T,X)]
        Y = np.random.uniform(0, 1, N)
        T = np.random.uniform(0, 1, N)
        right = [[[t],[1],[y]] for t,y in zip(T,Y)]

        pass

    def train(self, num_iter, len_data_set):
        opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
        
        boundary_data, init_data, int_data = self.generate_data_sets(len_data_set, 1)

        l = []
        for _ in tqdm(range(num_iter)):
            i = np.random.randint(len_data_set)
            g, val = self.gradients(boundary_data[i], init_data[i], int_data[i])
            opt.apply_gradients(zip(g, self.trainable_variables))
            l.append(val)
        return l

            
    ##################################################################
    

if __name__ == "__main__":
    pass