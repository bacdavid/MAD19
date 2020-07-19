# imports
import tensorflow as tf
import numpy as np

# placeholders
x = tf.placeholder(shape=[], dtype=tf.float32)
y_target = tf.placeholder(shape=[], dtype=tf.float32)

# weights
init_w = tf.random_normal(shape=[], stddev=0.1)
w = tf.Variable( init_w )

# bias
init_b = tf.constant(shape=[], value=0.1)
b = tf.Variable( init_b )

# model
y =  w * x + b

# loss
loss = 2 * tf.nn.l2_loss(y_target - y)

# optimizer for gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# tf session
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

# initializer
init = tf.global_variables_initializer()
sess.run(init)

# data
X = np.arange(-10., 10., 0.1)
Y = 5. * X + 10.

# training
for epoch in range(10000):
    idx = np.random.randint(0, len(X))
    final_error, _ = sess.run([loss, optimizer], feed_dict={x: X[idx], y_target: Y[idx]})

# read out
print('final error:  ' + str(final_error))
print('weight:  ' + str(sess.run(w)))
print('bias:  ' + str(sess.run(b)))