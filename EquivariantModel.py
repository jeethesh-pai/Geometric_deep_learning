import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
import numpy as np


class CustomHiddenDense(Layer):
    def __init__(self, activation="relu"):
        super(CustomHiddenDense, self).__init__()
        self.a = tf.Variable(np.random.normal(), name='a', trainable=True)
        self.b = tf.Variable(np.random.normal(), name='b', trainable=True)
        self.c = tf.Variable(np.random.normal(), name='c', trainable=True)
        self.w = [[self.a, self.b, self.c, self.b],
                  [self.b, self.a, self.b, self.c],
                  [self.c, self.b, self.a, self.b],
                  [self.b, self.c, self.b, self.a]]
        self.activation = activation

    def __call__(self, inputs):
        if self.activation == 'relu':
            return tf.nn.relu(tf.matmul(inputs, self.w))
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(tf.matmul(inputs, self.w))
        elif self.activation == 'tanh':
            return tf.nn.tanh(tf.matmul(inputs, self.w))
        else:
            return tf.matmul(inputs, self.w)


class OutputDense(Layer):
    def __init__(self):
        super(OutputDense, self).__init__()
        self.d = tf.Variable(np.random.normal(), name='d', trainable=True)
        self.w = [[self.d, -self.d],
                  [-self.d, -self.d],
                  [-self.d, self.d],
                  [self.d, self.d]]

    def __call__(self, inputs):
        return tf.matmul(inputs, self.w)


class EquiVariantModel(Model):
    def get_config(self):
        pass

    def call(self, inputs, training=True, mask=None):
        x = self.layer_list[0](inputs)
        for num in range(1, self.num_layers + 1):
            x = self.layer_list[num](x)
        return x

    def __init__(self, num_layers: int, activation: str):
        super(EquiVariantModel, self).__init__()
        self.layer_list = []
        for num in range(num_layers):
            self.layer_list.append(CustomHiddenDense(activation))
        self.num_layers = num_layers
        self.layer_list.append(OutputDense())

    def train_step(self, data: tuple):  # in our case the data is receiving fit(x, y, ...) therefore expecting a tuple
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss}


# a = tf.Variable(1., name='a', trainable=True)
# b = tf.Variable(1., name='b', trainable=True)
# c = tf.Variable(1., name='c', trainable=True)
# d = tf.Variable(1., name='d', trainable=True)
# w = np.array([[a, b], [c, b], [a, d]])
# print(w)
# x = [[1., 2., 3.]]
# with tf.GradientTape(persistent=True) as tape:
#     y = x @ w
#     loss = y
# print(tape.gradient(loss, [w, a, b, c, d]))
# print('something')
#
#
# a = tf.Variable(1., name='a', trainable=True)
# b = tf.Variable(1., name='b', trainable=True)
# c = tf.Variable(1., name='c', trainable=True)
# d = tf.Variable(1., name='d', trainable=True)
# # w = [[a, b], [c, b], [a, d]]
# w = [[a, b, c, b], [b, a, b, c], [c, b, a, b], [b, c, b, a]]
# x = tf.constant([[1., 2., 3., 4.]])
# with tf.GradientTape(persistent=True) as tape:
#     y = x @ w
#     loss = y
# print(*tape.gradient(loss, [w, a, b, c]), sep='\n')



