import numpy as np
import os
import random
import plotly.graph_objects as go
import plotly.express as px
from Network import Network
import tensorflow as tf
from heat1D_ex1 import HeatModel

seed = 2727
os.environ['PYTHONHASHSEED']=str(seed)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)



def true_solution(tx):
    return np.exp(-tx[...,0]) * np.sin(tx[...,1] * np.pi)

T = np.arange(0, 1 + 0.1, 0.01)
X = np.arange(-1, 1, 0.01)
grid = np.array([[t,x] for t in T for x in X])

W = true_solution(grid).reshape((len(T), len(X)))

true_solution = go.Figure(data=[go.Surface(z=W, x=X, y=T)])

true_solution.update_layout(title='True solution', margin=dict(l=65, r=50, b=65, t=90))

true_solution.show()




model = HeatModel()

Z = model.model(grid).numpy().reshape((len(T),len(X)))

pinn_plot = go.Figure(data=[go.Surface(z=Z, x=X, y=T)])

pinn_plot.update_layout(title='PINN plot', margin=dict(l=65, r=50, b=65, t=90))

pinn_plot.show()

# batch_size
n = 500
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

l = model.fit_SGD(data_int, data_init, data_left, data_right)


fig = px.line(l)
fig.show()

Z = model.model(grid).numpy().reshape((len(T),len(X)))
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T)])
fig.update_layout(title='PINN solution', margin=dict(l=65, r=50, b=65, t=90))
fig.show()