import tensorflow as tf
import numpy as np

input_x = tf.placeholder(tf.float32, [100, 1], name='input vector')
represetnt_z = tf.Variable(shape=[100, 100], initial_value=tf.random_normal_initializer(), name='represent z')
output_y = tf.matmul(represetnt_z, input_x, name='output vector')
optimizer = tf.reduce_sum(tf.pow(tf.subtract(output_y, input_x), 2))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
