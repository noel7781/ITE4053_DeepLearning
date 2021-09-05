import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

learning_rate = 0.001
training_epochs = 50
batch_size = 128
regularization_strength = 0.01

class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._loss = float('inf')
        self._step = 0
        self.patience = patience
        self.verbose = verbose
    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print("Early stopped!")
                return True
        else:
            self._loss = loss
            self._step = 0
        return False

def batch_data(shuffled_idx, batch_size, data, labels, start_idx):
    idx = shuffled_idx[start_idx:start_idx+batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def build_CNN_classifier(x):
    x_image = x

    W1 = tf.get_variable(name="W1", shape=[3, 3, 3, 64], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(name="b1", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
    c1 = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME')
    l1 = tf.nn.relu(tf.nn.bias_add(c1, b1))

    W2 = tf.get_variable(name="W2", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name="b2", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
    c2 = tf.nn.conv2d(l1, W2, strides=[1, 1, 1, 1], padding='SAME')
    l2 = tf.nn.relu(tf.nn.bias_add(c2, b2))
    l2_pool = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W3 = tf.get_variable(name="W3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable(name="b3", shape=[128], initializer=tf.contrib.layers.xavier_initializer())
    c3 = tf.nn.conv2d(l2_pool, W3, strides=[1, 1, 1, 1], padding='SAME')
    l3 = tf.nn.relu(tf.nn.bias_add(c3, b3))

    W4 = tf.get_variable(name="W4", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable(name="b4", shape=[128], initializer=tf.contrib.layers.xavier_initializer())
    c4 = tf.nn.conv2d(l3, W4, strides=[1, 1, 1, 1], padding='SAME')
    l4 = tf.nn.relu(tf.nn.bias_add(c4, b4))
    l4_pool = tf.nn.max_pool(l4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    W5 = tf.get_variable(name="W5", shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable(name="b5", shape=[256], initializer=tf.contrib.layers.xavier_initializer())
    c5 = tf.nn.conv2d(l4_pool, W5, strides=[1, 1, 1, 1], padding='SAME')
    l5 = tf.nn.relu(tf.nn.bias_add(c5, b5))

    W6 = tf.get_variable(name="W6", shape=[3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.get_variable(name="b6", shape=[256], initializer=tf.contrib.layers.xavier_initializer())
    c6 = tf.nn.conv2d(l5, W6, strides=[1, 1, 1, 1], padding='SAME')
    l6 = tf.nn.relu(tf.nn.bias_add(c6, b6))
    l6_pool = tf.nn.max_pool(l6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W7 = tf.get_variable(name="W7", shape=[3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
    b7 = tf.get_variable(name="b7", shape=[512], initializer=tf.contrib.layers.xavier_initializer())
    c7 = tf.nn.conv2d(l6_pool, W7, strides=[1, 1, 1, 1], padding='SAME')
    l7 = tf.nn.relu(tf.nn.bias_add(c7, b7))

    W8 = tf.get_variable(name="W8", shape=[3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    b8 = tf.get_variable(name="b8", shape=[512], initializer=tf.contrib.layers.xavier_initializer())
    c8 = tf.nn.conv2d(l7, W8, strides=[1, 1, 1, 1], padding='SAME')
    l8 = tf.nn.relu(tf.nn.bias_add(c8, b8))

    W9 = tf.get_variable(name="W9", shape=[3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    b9 = tf.get_variable(name="b9", shape=[512], initializer=tf.contrib.layers.xavier_initializer())
    c9 = tf.nn.conv2d(l8, W9, strides=[1, 1, 1, 1], padding='SAME')
    l9 = tf.nn.relu(tf.nn.bias_add(c9, b9))

    l9_pool = tf.nn.max_pool(l9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    l9_flat = tf.reshape(l9_pool, [-1, 2*2*512])

    W_fc1 = tf.get_variable(name="W_fc1", shape=[2*2*512, 2*2*512], initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.get_variable(name="b_fc1", shape=[2*2*512], initializer=tf.contrib.layers.xavier_initializer())
    l10 = tf.nn.bias_add(tf.matmul(l9_flat, W_fc1), b_fc1)
    l10 = tf.nn.dropout(l10, keep_prob)
    W_fc2 = tf.get_variable(name="W_fc2", shape=[2*2*512, 2*2*512], initializer=tf.contrib.layers.xavier_initializer())
    b_fc2 = tf.get_variable(name="b_fc2", shape=[2*2*512], initializer=tf.contrib.layers.xavier_initializer())
    l11 = tf.nn.bias_add(tf.matmul(l10, W_fc2), b_fc2)
    l11 = tf.nn.dropout(l10, keep_prob)
    W_fc3 = tf.get_variable(name="W_fc3", shape=[2*2*512, 10], initializer=tf.contrib.layers.xavier_initializer())
    b_fc3 = tf.get_variable(name="b_fc3", shape=[10], initializer=tf.contrib.layers.xavier_initializer())
    logits = tf.nn.bias_add(tf.matmul(l11, W_fc3), b_fc3)
    hypothesis = tf.nn.softmax(logits)

    return hypothesis, logits

np.random.seed(51)
tf.set_random_seed(51)
early_stopping = EarlyStopping(patience=5, verbose=1)

ckpt_path = "output/"

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder_with_default(1.0, shape=())

x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")

dev_num = len(x_train) // 4

x_dev = x_train[:dev_num]
y_dev = y_train[:dev_num]

x_train = x_train[dev_num:]
y_train = y_train[dev_num:]

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_dev_one_hot = tf.squeeze(tf.one_hot(y_dev, 10),axis=1)

y_pred, logits = build_CNN_classifier(x)

regularization_loss = regularization_strength * tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables() if len(v.shape) > 1]) #가중치 규제 (L2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)) + regularization_loss
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

total_batch = int(len(x_train)/batch_size) if len(x_train)%batch_size == 0 else int(len(x_train)/batch_size) + 1


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("학습시작")

    for epoch in range(training_epochs):
        print("Epoch", epoch+1)
        start = 0
        shuffled_idx = np.arange(0, len(x_train))
        np.random.shuffle(shuffled_idx)
        losses = 0

        for i in range(total_batch):
            batch = batch_data(shuffled_idx, batch_size, x_train, y_train_one_hot.eval(), i*batch_size)
            _, loss = sess.run([train_step, cost], feed_dict={x: batch[0], y: batch[1], keep_prob:0.6})
            losses += loss
        print("losses:", losses)
            
        if early_stopping.validate(losses):
            break
    
    saver = tf.train.Saver()
    saver.save(sess, ckpt_path)
    saver.restore(sess, ckpt_path)

    y_prediction = np.argmax(y_pred.eval(feed_dict={x: x_dev, keep_prob:1}), 1)
    y_true = np.argmax(y_dev_one_hot.eval(), 1)
    dev_f1 = f1_score(y_true, y_prediction, average="weighted") # f1 스코어 측정
    print("dev 데이터 f1 score: %f" % dev_f1)

    # 밑에는 건드리지 마세요
    x_test = np.load("data/x_test.npy")
    test_logits = y_pred.eval(feed_dict={x: x_test})
    np.save("result", test_logits)
