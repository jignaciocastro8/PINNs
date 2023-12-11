import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import random
import numpy as np
sys.path.append('D:/Math PhD/PINNs code and papers/PINNs')
import scipy.io   
import plotly.graph_objects as go
from BugersModelEx1 import BurgersModel
import tensorflow as tf
import plotly.express as px
import time

seed = 2727
os.environ['PYTHONHASHSEED']=str(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


path = 'D:/Math PhD/PINNs code and papers/PINNs/BurgersEquation/Example1/Burgers.mat'
data = scipy.io.loadmat(path) 
X = data['x'][...,0]                                   
T = data['t'][...,0]                                   
usol = data['usol'].T

fig = go.Figure(data=[go.Surface(z=usol, x=X, y=T)])
fig.update_layout(
    title='Given data (.mat)',
    scene = dict(xaxis = dict(title='space', nticks=4, range=[-1,1],),
                 yaxis = dict(title='time', nticks=4, range=[0,1])))
fig.show()

############################### hyperparameters ###############################

# Model architecture
layers = [2,32,32,32,32,32,32,1]
# PDE loss weight
mu = 0.01
# Learning rate
learning_rate = 0.001
# Amount of data points
n = 600
# Mini batch
mini_batch = 200
# (Stochastic) gradient descent iterations
opti_iter = 3000

###############################################################################

# Interior data t \in [0,1], x \in [-1,1]
data_int = np.random.rand(n, 2)
data_int[..., 1] = data_int[..., 1]*2 - 1
data_int = tf.Variable(data_int, trainable=False, dtype="float32")
# t=0 data
data_init = np.random.rand(n,2) * 2 - 1
data_init[..., 0] = 0
data_init = tf.constant(data_init, dtype="float32")
# x = -1 data
data_left = np.random.rand(n,2) 
data_left[..., 1] = -1
data_left = tf.constant(data_left, dtype="float32")
# x = 1 data
data_right = np.random.rand(n,2)
data_right[..., 1] = 1
data_right = tf.constant(data_right, dtype="float32")

model = BurgersModel(architecture=layers, 
                     mu=mu, 
                     learning_rate=learning_rate, 
                     opti_iter=opti_iter, 
                     mini_batch=mini_batch)

model.model.summary()

start = time.time()
l = model.fit_Adam(data_int, data_init, data_left, data_right)
end = time.time()
print('Time: ', end - start)


new_l = [l[i].numpy() for i in range(opti_iter)]
#new_l = l.concat()
fig = px.line(new_l, log_y=True)
fig.update_layout(
    xaxis_title="Epoch",
    yaxis_title="Loss",
    font=dict(
        family="Courier New, condensed",
        size=18,
    )
)
fig.show()

time = np.arange(0, 1, 0.001)
space = np.arange(-1, 1, 0.001)
grid = np.array([[t,x] for t in time for x in space])

Z = model.model(grid).numpy().reshape((len(time), len(space)))
fig = go.Figure(data=[go.Surface(z=Z, x=space, y=time)])
fig.update_layout(
    title='PINN solution for Burgers equation',
    scene = dict(xaxis = dict(title='space', nticks=4, range=[-1,1],),
                 yaxis = dict(title='time', nticks=4, range=[0,1])))
fig.show()

