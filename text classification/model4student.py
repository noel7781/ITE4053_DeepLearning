import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn_cell

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

def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1 # plus the 0th word

def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]

def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])

def build_classifier(x, vocabulary_size, EMBEDDING_DIM, HIDDEN_SIZE):
    # Embedding layer
    embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, x)

    '''
    cells = [tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE) for _ in range(1)]
    rnn_cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob, state_keep_prob=keep_prob) for cell in cells]
    rnn_cells = tf.nn.rnn_cell.MultiRNNCell(rnn_cells)
    rnn_outputs, states = tf.nn.dynamic_rnn(cell=rnn_cells, inputs=batch_embedded, dtype=tf.float32)
    '''
    cell1 = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE, name="c1")

    # RNN layer
    rnn_outputs1, states1 = tf.nn.dynamic_rnn(cell=cell1, inputs=batch_embedded, dtype=tf.float32)
    rnn_outputs = rnn_outputs1

    w_omega = tf.Variable(tf.random_normal([HIDDEN_SIZE, ATTENTION_SIZE], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([ATTENTION_SIZE], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([ATTENTION_SIZE], stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(rnn_outputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')

        output = tf.reduce_sum(rnn_outputs * tf.expand_dims(alphas, -1), 1)

    # Fully connected layer
    W = tf.Variable(tf.random_uniform([HIDDEN_SIZE, 2], -1.0, 1.0), trainable=True)
    b = tf.Variable(tf.random_uniform([2], -1.0, 1.0), trainable=True)
    #logits = tf.nn.bias_add(tf.matmul(rnn_outputs[:,-1], W), b)
    logits = tf.nn.bias_add(tf.matmul(output, W), b)
    hypothesis = tf.nn.softmax(logits)

    return hypothesis, logits

ckpt_path = "output/"

SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 80
HIDDEN_SIZE = 128
BATCH_SIZE = 256
NUM_EPOCHS = 50
learning_rate = 0.001
ATTENTION_SIZE = EMBEDDING_DIM

regularization_strength = 0.01
keep_prob = tf.placeholder_with_default(1.0, shape=())
early_stopping = EarlyStopping(patience=5, verbose=1)

# Load the data set
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
x_test = np.load("data/x_test.npy")

np.load = np_load_old


dev_num = len(x_train) // 4

x_dev = x_train[:dev_num]
y_dev = y_train[:dev_num]

x_train = x_train[dev_num:]
y_train = y_train[dev_num:]

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 2))
y_dev_one_hot = tf.squeeze(tf.one_hot(y_dev, 2))


# Sequences pre-processing
vocabulary_size = get_vocabulary_size(x_train)
x_dev = fit_in_vocabulary(x_dev, vocabulary_size)
x_train = zero_pad(x_train, SEQUENCE_LENGTH)
x_dev = zero_pad(x_dev, SEQUENCE_LENGTH)

batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')

y_pred, logits = build_classifier(batch_ph, vocabulary_size, EMBEDDING_DIM, HIDDEN_SIZE)

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_ph, logits=logits))
regularization_loss = regularization_strength * tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables() if len(v.shape) > 1]) #가중치 규제 (L2)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_ph, logits=logits)) + regularization_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Accuracy metric
is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(target_ph, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

total_batch = int(len(x_train)/BATCH_SIZE) if len(x_train)%BATCH_SIZE == 0 else int(len(x_train)/BATCH_SIZE) + 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("학습시작")

    min_loss = 1e9
    for epoch in range(NUM_EPOCHS):
        print("Epoch", epoch + 1)
        start = 0
        shuffled_idx = np.arange(0, len(x_train))
        np.random.shuffle(shuffled_idx)

        for i in range(total_batch):
            batch = batch_data(shuffled_idx, BATCH_SIZE, x_train, y_train_one_hot.eval(), i * BATCH_SIZE)
            acc_val, loss_val, _ = sess.run([accuracy, loss, optimizer], feed_dict={batch_ph: batch[0], target_ph: batch[1], keep_prob:0.5})
        if early_stopping.validate(loss_val):
            break
        saver = tf.train.Saver()
        if min_loss > loss_val:
            dev_accuracy = accuracy.eval(feed_dict={batch_ph: x_dev, target_ph: np.asarray(y_dev_one_hot.eval())})
            print("Acc:",acc_val, " and loss:", loss_val)
            print("dev Accuracy: %f Saved" % dev_accuracy)
            min_loss = loss_val
            saver.save(sess, ckpt_path)
        saver.restore(sess, ckpt_path)

    dev_accuracy = accuracy.eval(feed_dict={batch_ph: x_dev, target_ph: np.asarray(y_dev_one_hot.eval())})
    print("dev 데이터 Accuracy: %f" % dev_accuracy)

    # 밑에는 건드리지 마세요
    x_test = fit_in_vocabulary(x_test, vocabulary_size)
    x_test = zero_pad(x_test, SEQUENCE_LENGTH)

    test_logits = y_pred.eval(feed_dict={batch_ph: x_test})
    np.save("result", test_logits)
