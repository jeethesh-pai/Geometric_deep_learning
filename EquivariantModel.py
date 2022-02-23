"""
This Script describes two ways of defining an Equi-variant Neural network model. First way using the declaration of
tf.Variable for the weights (see CustomHiddenDense Class for details), second one using layer.add_weight() and
assembling a matrix according to dlr_worksheet.pdf later (see CustomHiddenDense2 Class). However Experiments showed us
that using layer.add_weights() method produces stable results than using tf.Variable
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import numpy as np


class CustomHiddenDense(Layer):
    """
    Defines a layer class according to the Equi-variant model description using tf.Variable concept
    :param activation: Activation function to be applied on the output
    """
    def __init__(self, activation="relu"):
        super(CustomHiddenDense, self).__init__()
        self.a = tf.Variable(np.random.normal(), name='a', trainable=True, dtype=tf.float64)
        self.b = tf.Variable(np.random.normal(), name='b', trainable=True, dtype=tf.float64)
        self.c = tf.Variable(np.random.normal(), name='c', trainable=True, dtype=tf.float64)
        self.activation = activation

    def __call__(self, inputs):
        """
        multiplies the layer kernel with the inputs
        :param inputs: tensor of shape [None, 4]
        :return: tensor of shape [None, 4]
        """
        w = [[self.a, self.b, self.c, self.b],
             [self.b, self.a, self.b, self.c],
             [self.c, self.b, self.a, self.b],
             [self.b, self.c, self.b, self.a]]
        if self.activation == 'relu':
            return tf.nn.relu(tf.matmul(inputs, w))
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(tf.matmul(inputs, w))
        elif self.activation == 'tanh':
            return tf.nn.tanh(tf.matmul(inputs, w))
        else:
            return tf.matmul(inputs, w)


class CustomHiddenDense2(Layer):
    """
    Defines a layer class according to the Equi-variant model description using add_weight() concept
    :param activation: Activation function to be applied on the output
    """
    def __init__(self, activation="relu"):
        super(CustomHiddenDense2, self).__init__()
        self.activation = activation
        initializer = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.05)
        self.a = self.add_weight(shape=None, initializer=initializer, trainable=True, name='a')
        self.b = self.add_weight(shape=None, initializer=initializer, trainable=True, name='b')
        self.c = self.add_weight(shape=None, initializer=initializer, trainable=True, name='c')

    def __call__(self, inputs):
        """
        multiplies the layer kernel with the inputs
        :param inputs: tensor of shape [None, 4]
        :return: tensor of shape [None, 4]
        """
        w = [[self.a, self.b, self.c, self.b],
             [self.b, self.a, self.b, self.c],
             [self.c, self.b, self.a, self.b],
             [self.b, self.c, self.b, self.a]]
        if self.activation == 'relu':
            return tf.nn.relu(tf.matmul(inputs, w))
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(tf.matmul(inputs, w))
        elif self.activation == 'tanh':
            return tf.nn.tanh(tf.matmul(inputs, w))
        else:
            return tf.matmul(inputs, w)


class OutputDense2(Layer):
    """
    Output layer which created the output predicted tensor using add_weight concept
    """
    def __init__(self):
        super(OutputDense2, self).__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.05)
        self.d = self.add_weight(shape=None, initializer=initializer, trainable=True, name='d')

    def __call__(self, inputs):
        """
        multiplies the layer kernel with the inputs
        :param inputs: tensor of shape [None, 4]
        :return: tensor of shape [None, 2]
        """
        w = [[self.d, -self.d],
             [-self.d, -self.d],
             [-self.d, self.d],
             [self.d, self.d]]
        return tf.matmul(inputs, w)


class OutputDense(Layer):
    """
    Output layer which created the output predicted tensor using tf.Variable concept
    """
    def __init__(self):
        super(OutputDense, self).__init__()
        self.d = tf.Variable(np.random.normal(), name='d', trainable=True, dtype=tf.float64)

    def __call__(self, inputs):
        """
        multiplies the layer kernel with the inputs
        :param inputs: tensor of shape [None, 4]
        :return: tensor of shape [None, 2]
        """
        w = [[self.d, -self.d],
             [-self.d, -self.d],
             [-self.d, self.d],
             [self.d, self.d]]
        return tf.matmul(inputs, w)


class EquiVariantModel(Model):
    def get_config(self):
        pass

    def call(self, inputs, training=True, mask=None):
        x = self.layer_list[0](inputs)
        for num in range(1, self.num_layers + 1):
            x = self.layer_list[num](x)
        return x

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None,
                run_eagerly=False, steps_per_execution=None, **kwargs):
        """
        compiles the model
        :param optimizer: Optimizer to be used for gradient descent optimization ('rmsprop', 'adam', default: sgd)
        :param loss: loss function (currently only mse accepted)
        :param metrics: not valid for this class
        :param loss_weights:
        :param weighted_metrics:
        :param run_eagerly:
        :param steps_per_execution:
        :param kwargs:
        :return:
        """
        if optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop()
        elif optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam()
        else:
            optimizer = tf.keras.optimizers.SGD()
        if loss == 'mse':
            loss = tf.keras.losses.MeanSquaredError(reduction='auto')
        else:
            print('loss function not identified')
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metrics
        super().compile(loss=loss, optimizer=optimizer, run_eagerly=run_eagerly)

    def __init__(self, num_layers: int, activation: str):
        """
        creates an Equi-variant model
        :param num_layers: number of feed forward layers to be created for this model
        :param activation: activation to be used for all the layers
        """
        super(EquiVariantModel, self).__init__()
        self.layer_list = []
        for num in range(num_layers):
            self.layer_list.append(CustomHiddenDense2(activation))
        self.num_layers = num_layers
        self.layer_list.append(OutputDense2())

    @tf.function()
    def train_step(self, data: tuple):  # in our case the data is receiving fit(x, y, ...) therefore expecting a tuple
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred)
        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss}

    def test_step(self, data: tuple):
        x, y = data
        y_pred = self(x, training=False)  # Forward pass
        loss = self.compiled_loss(y, y_pred)
        return {"loss": loss}


# check the gradient flow for both the methods
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
