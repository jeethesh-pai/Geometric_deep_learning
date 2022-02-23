# import numpy as np
#
#
# a = 2
# b = 3
# c = 4
# d = 1
#
# w1 = np.array([[a, b, c, b],
#                [b, a, b, c],
#                [c, b, a, b],
#                [b, c, b, a]])
#
# w2 = np.array([[d, -d],
#                [-d, -d],
#                [-d, d],
#                [d, d]])
#
# x = np.array([4, 3, 2, 1])[np.newaxis, :]
# y = x @ w1 @ w2
# print(y)
import tensorflow as tf
import numpy as np


a = tf.Variable(np.random.normal(), name='a', trainable=True, dtype=tf.float64)
b = tf.Variable(np.random.normal(), name='b', trainable=True, dtype=tf.float64)
c = tf.Variable(np.random.normal(), name='c', trainable=True, dtype=tf.float64)
d = tf.Variable(np.random.normal(), name='d', trainable=True, dtype=tf.float64)

x = tf.constant([[4, 3, 2, 1]], dtype=tf.float64)
w1 = [[a, b, c, b],
      [b, a, b, c],
      [c, b, a, b],
      [b, c, b, a]]
w2 = [[d, -d],
      [-d, -d],
      [-d, d],
      [d, d]]
y_act = tf.constant([[1, 2]], dtype=tf.float64)
with tf.GradientTape(persistent=True) as tape:
    y_pred = x @ w1 @ w2
    loss = (y_act - y_pred)**2
gradients2 = tape.gradient(loss, d)
gradients1 = tape.gradient(loss, [a, b, c])

print(*gradients2, sep='\n')
