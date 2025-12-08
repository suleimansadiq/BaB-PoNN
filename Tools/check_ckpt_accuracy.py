#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.keras.datasets import mnist

# === Config ===
CKPT_PATH = "posit8_ultratinymlp.ckpt"   # checkpoint prefix (no .meta/.index)

# Load MNIST
(_, _), (X_test, y_test) = mnist.load_data()
X_test = ((X_test.astype(np.float32) - 127.5) / 127.5)   # normalize [-1,1]
X_test = np.expand_dims(X_test, axis=3)                  # shape (10000,28,28,1)

# Placeholders
x = tf.placeholder(tf.float32, (None,28,28,1), name="inputs")
y = tf.placeholder(tf.int32,   (None),        name="labels")

# Rebuild same tiny MLP (float shadow version)
def UltraTinyMLP(x):
    x_flat = tf.reshape(x, [-1, 28*28])
    W1 = tf.get_variable("W1_master", shape=(784,8), dtype=tf.float32)
    b1 = tf.get_variable("b1_master", shape=(8,), dtype=tf.float32)
    W2 = tf.get_variable("W2_master", shape=(8,10), dtype=tf.float32)
    b2 = tf.get_variable("b2_master", shape=(10,), dtype=tf.float32)

    h1 = tf.matmul(x_flat, W1) + b1
    h1 = tf.nn.relu(h1)
    logits = tf.matmul(h1, W2) + b2
    return logits

logits = UltraTinyMLP(x)
correct = tf.equal(tf.argmax(logits,1,output_type=tf.int32), y)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver()

# Run evaluation
with tf.Session() as sess:
    saver.restore(sess, CKPT_PATH)
    acc_val = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
    print("Checkpoint accuracy on MNIST test set (no quantization): {:.4f}".format(acc_val))
