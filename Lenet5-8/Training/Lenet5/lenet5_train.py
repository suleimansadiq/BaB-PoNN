# lenet5_shadow_export_p8.py
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.datasets import mnist
import sys, csv, os, time, json

# Repro (optional)
np.random.seed(1)
tf.set_random_seed(2)

# -----------------------------
# Args / dtype setup
# -----------------------------
if len(sys.argv) > 1:
    data_t = sys.argv[1]
else:
    data_t = 'posit8'  # default to posit8 export

if   data_t == 'posit32':
    eps = 1e-8;  posit = np.posit32; tf_type = tf.posit32
elif data_t == 'posit16':
    eps = 1e-4;  posit = np.posit16; tf_type = tf.posit16
elif data_t == 'posit8':
    eps = 1e-5;  posit = np.posit8;  tf_type = tf.float32  # masters in f32, forward casts
elif data_t == 'float16':
    eps = 1e-4;  posit = np.float16; tf_type = tf.float16
elif data_t == 'float32':
    eps = 1e-8;  posit = np.float32; tf_type = tf.float32
else:
    eps = 1e-8;  data_t = 'float32'; posit = np.float32; tf_type = tf.float32

# -----------------------------
# Hyperparams
# -----------------------------
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.001

print("Type:", data_t, "EPOCHS:", EPOCHS, "BATCH_SIZE:", BATCH_SIZE, flush=True)

# -----------------------------
# Data (keep raw u8 aside for export)
# -----------------------------
(X_train_u8, y_train), (X_test_u8, y_test) = mnist.load_data()
X_train_u8 = X_train_u8.astype(np.uint8)
X_test_u8  = X_test_u8.astype(np.uint8)

# Train/eval arrays (normalized to [-1,1])
X_train = np.expand_dims(X_train_u8, axis=3)
X_test  = np.expand_dims(X_test_u8,  axis=3)
X_train = ((X_train - 127.5) / 127.5).astype(posit)
X_test  = ((X_test  - 127.5) / 127.5).astype(posit)

# Pad to 32×32 for LeNet (compute graph uses 32×32)
X_train_32 = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test_32  = np.pad(X_test,  ((0,0),(2,2),(2,2),(0,0)), 'constant')

# Feed dtype for placeholders
if data_t == 'posit8':
    X_train_feed = X_train_32.astype(np.float32)
    X_test_feed  = X_test_32.astype(np.float32)
else:
    X_train_feed = X_train_32
    X_test_feed  = X_test_32

print("Train:", X_train_feed.shape, "Test:", X_test_feed.shape, flush=True)

# -----------------------------
# Shadow-cast helper (master f32 -> posit8 -> back f32)
# -----------------------------
def q_cast_fwd(var_f32, name=None):
    v8  = tf.cast(var_f32, tf.posit8, name=(name+"_to_p8") if name else None)
    v32 = tf.cast(v8, tf.float32,     name=(name+"_back_f32") if name else None)
    return v32

# -----------------------------
# Model (LeNet-5, no BN, biases included)
# -----------------------------
def conv2d(x, W, stride=1, padding='VALID'):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding=padding)

def LeNet(x):
    mu = 0.0; sigma = 0.1

    if data_t != 'posit8':
        # vars in requested tf_type
        conv1_W = tf.Variable(tf.truncated_normal((5,5,1,6), mean=mu, stddev=sigma, dtype=tf_type), name='conv1_W')
        conv1_b = tf.Variable(tf.zeros((6,), dtype=tf_type), name='conv1_b')
        conv2_W = tf.Variable(tf.truncated_normal((5,5,6,16), mean=mu, stddev=sigma, dtype=tf_type), name='conv2_W')
        conv2_b = tf.Variable(tf.zeros((16,), dtype=tf_type), name='conv2_b')
        fc1_W   = tf.Variable(tf.truncated_normal((400,120), mean=mu, stddev=sigma, dtype=tf_type), name='fc1_W')
        fc1_b   = tf.Variable(tf.zeros((120,), dtype=tf_type), name='fc1_b')
        fc2_W   = tf.Variable(tf.truncated_normal((120,84), mean=mu, stddev=sigma, dtype=tf_type), name='fc2_W')
        fc2_b   = tf.Variable(tf.zeros((84,), dtype=tf_type), name='fc2_b')
        fc3_W   = tf.Variable(tf.truncated_normal((84,10), mean=mu, stddev=sigma, dtype=tf_type), name='fc3_W')
        fc3_b   = tf.Variable(tf.zeros((10,), dtype=tf_type), name='fc3_b')

        y = conv2d(x, conv1_W, 1, 'VALID') + conv1_b
        y = tf.nn.relu(y)
        y = tf.nn.max_pool(y, [1,2,2,1], [1,2,2,1], 'VALID')  # 14×14×6
        y = conv2d(y, conv2_W, 1, 'VALID') + conv2_b
        y = tf.nn.relu(y)
        y = tf.nn.max_pool(y, [1,2,2,1], [1,2,2,1], 'VALID')  # 5×5×16
        flat = tf.contrib.layers.flatten(y)                   # 400
        y = tf.matmul(flat, fc1_W) + fc1_b; y = tf.nn.relu(y)
        y = tf.matmul(y,   fc2_W) + fc2_b; y = tf.nn.relu(y)
        logits = tf.matmul(y, fc3_W) + fc3_b
        return logits

    # posit8: masters in float32, cast through p8 in forward
    conv1_Wm = tf.Variable(tf.truncated_normal((5,5,1,6), mean=mu, stddev=sigma, dtype=tf.float32), name='conv1_W_master')
    conv1_bm = tf.Variable(tf.zeros((6,), dtype=tf.float32), name='conv1_b_master')
    conv2_Wm = tf.Variable(tf.truncated_normal((5,5,6,16), mean=mu, stddev=sigma, dtype=tf.float32), name='conv2_W_master')
    conv2_bm = tf.Variable(tf.zeros((16,), dtype=tf.float32), name='conv2_b_master')
    fc1_Wm   = tf.Variable(tf.truncated_normal((400,120), mean=mu, stddev=sigma, dtype=tf.float32), name='fc1_W_master')
    fc1_bm   = tf.Variable(tf.zeros((120,), dtype=tf.float32), name='fc1_b_master')
    fc2_Wm   = tf.Variable(tf.truncated_normal((120,84), mean=mu, stddev=sigma, dtype=tf.float32), name='fc2_W_master')
    fc2_bm   = tf.Variable(tf.zeros((84,), dtype=tf.float32), name='fc2_b_master')
    fc3_Wm   = tf.Variable(tf.truncated_normal((84,10), mean=mu, stddev=sigma, dtype=tf.float32), name='fc3_W_master')
    fc3_bm   = tf.Variable(tf.zeros((10,), dtype=tf.float32), name='fc3_b_master')

    conv1_W = q_cast_fwd(conv1_Wm, 'conv1_W'); conv1_b = q_cast_fwd(conv1_bm, 'conv1_b')
    conv2_W = q_cast_fwd(conv2_Wm, 'conv2_W'); conv2_b = q_cast_fwd(conv2_bm, 'conv2_b')
    fc1_W   = q_cast_fwd(fc1_Wm,   'fc1_W');   fc1_b   = q_cast_fwd(fc1_bm,   'fc1_b')
    fc2_W   = q_cast_fwd(fc2_Wm,   'fc2_W');   fc2_b   = q_cast_fwd(fc2_bm,   'fc2_b')
    fc3_W   = q_cast_fwd(fc3_Wm,   'fc3_W');   fc3_b   = q_cast_fwd(fc3_bm,   'fc3_b')

    y = conv2d(x, conv1_W, 1, 'VALID') + conv1_b
    y = tf.nn.relu(y)
    y = tf.nn.max_pool(y, [1,2,2,1], [1,2,2,1], 'VALID')  # 14×14×6
    y = conv2d(y, conv2_W, 1, 'VALID') + conv2_b
    y = tf.nn.relu(y)
    y = tf.nn.max_pool(y, [1,2,2,1], [1,2,2,1], 'VALID')  # 5×5×16
    flat = tf.contrib.layers.flatten(y)                   # 400
    y = tf.matmul(flat, fc1_W) + fc1_b; y = tf.nn.relu(y)
    y = tf.matmul(y,   fc2_W) + fc2_b; y = tf.nn.relu(y)
    logits = tf.matmul(y, fc3_W) + fc3_b
    return logits

# -----------------------------
# Graph / training ops
# -----------------------------
ph_dtype = tf.float32 if data_t == 'posit8' else tf_type
x = tf.placeholder(ph_dtype, (None,32,32,1), name='inputs')
y = tf.placeholder(tf.int32, (None,),        name='labels')

logits = tf.identity(LeNet(x), name="logits")
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
if data_t == 'posit8':
    lr_val, eps_val = 0.001, 1e-5
else:
    lr_val, eps_val = np.cast[posit](LR), np.cast[posit](eps)

opt = tf.train.AdamOptimizer(learning_rate=lr_val, epsilon=eps_val)
train_op = opt.minimize(loss)

acc_op = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1,output_type=tf.int32), y), tf.float32))
saver = tf.train.Saver()

# -----------------------------
# Train
# -----------------------------
export_dir = './export_lenet5_p8/'
os.makedirs(export_dir, exist_ok=True)

print("\nStart Training", flush=True)
tic = time.time()
hist = {"loss":[], "acc":[], "val_loss":[], "val_acc":[]}

def validate(sess):
    n = len(X_test_feed)
    total_loss = 0.0; total_acc = 0.0
    for i in range(0, n, BATCH_SIZE):
        bx = X_test_feed[i:i+BATCH_SIZE]; by = y_test[i:i+BATCH_SIZE]
        lo, ac = sess.run([loss, acc_op], feed_dict={x:bx, y:by})
        total_loss += lo * len(bx); total_acc += ac * len(bx)
    return total_loss/n, total_acc/n

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ntr = len(X_train_feed)
    for ep in range(EPOCHS):
        X_train_feed, y_train = shuffle(X_train_feed, y_train)
        ep_loss = 0.0; ep_acc = 0.0
        for i in range(0, ntr, BATCH_SIZE):
            bx = X_train_feed[i:i+BATCH_SIZE]; by = y_train[i:i+BATCH_SIZE]
            _, lo, ac = sess.run([train_op, loss, acc_op], feed_dict={x:bx, y:by})
            ep_loss += lo * len(bx); ep_acc += ac * len(bx)
        ep_loss /= ntr; ep_acc /= ntr
        vloss, vacc = validate(sess)
        hist["loss"].append(ep_loss); hist["acc"].append(ep_acc)
        hist["val_loss"].append(vloss); hist["val_acc"].append(vacc)
        print("Epoch {}: Train Loss={:.4f} Train Acc={:.3f} | Val Loss={:.4f} Val Acc={:.3f}"
              .format(ep+1, ep_loss, ep_acc, vloss, vacc), flush=True)

    # -----------------------------
    # Export: MNIST test set (u8) & labels
    # -----------------------------
    # Flatten 28×28 to 784 (no pad). Row-major.
    X_test_u8.reshape(-1, 28*28).astype(np.uint8).tofile(os.path.join(export_dir, "mnist_images_u8.bin"))
    y_test.astype(np.uint8).tofile(os.path.join(export_dir, "mnist_labels_u8.bin"))

    # -----------------------------
    # Export: LeNet weights/biases as POSIT-8 raw bytes
    # -----------------------------
    # Helper to cast numpy float32 -> posit8 -> raw bytes
    def save_p8(path, arr_f32):
        arr_p8 = arr_f32.astype(np.float32).astype(np.posit8)   # 1 byte/elem custom dtype
        # Interpret the posit8 storage as bytes:
        arr_p8.view(np.uint8).tofile(path)

    # Grab tensors by names we set (masters if posit8; otherwise as-is)
    # For posit8 path, we read the *master* tensors, not the casted views.
    fetch = {}
    if data_t == 'posit8':
        names = [
            'conv1_W_master:0','conv1_b_master:0',
            'conv2_W_master:0','conv2_b_master:0',
            'fc1_W_master:0',  'fc1_b_master:0',
            'fc2_W_master:0',  'fc2_b_master:0',
            'fc3_W_master:0',  'fc3_b_master:0',
        ]
    else:
        names = [
            'conv1_W:0','conv1_b:0',
            'conv2_W:0','conv2_b:0',
            'fc1_W:0',  'fc1_b:0',
            'fc2_W:0',  'fc2_b:0',
            'fc3_W:0',  'fc3_b:0',
        ]
    tensors = [tf.get_default_graph().get_tensor_by_name(nm) for nm in names]
    vals = sess.run(tensors)

    keys = ['conv1_W','conv1_b','conv2_W','conv2_b','fc1_W','fc1_b','fc2_W','fc2_b','fc3_W','fc3_b']
    shapes = {}
    for k, v in zip(keys, vals):
        shapes[k] = list(v.shape)
        save_p8(os.path.join(export_dir, f"{k}_p8.bin"), v)

    # Save shapes & layout doc
    meta = {
        "layout": {
            "conv": "kernel layout [kh, kw, in_c, out_c], row-major contiguous",
            "fc":   "matrix layout [in_dim, out_dim], row-major contiguous",
            "dtype": "posit8 (1 byte per element) raw, no header"
        },
        "shapes": shapes
    }
    with open(os.path.join(export_dir, "tensor_shapes.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # (Optional) save training curves
    with open(os.path.join(export_dir, "lenet5_shadow_hist.csv"), "w", newline='') as f:
        w = csv.writer(f); ks = list(hist.keys()); w.writerow(ks)
        for row in zip(*[hist[k] for k in ks]): w.writerow(row)

print("Export complete:", export_dir, flush=True)

