import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.get_variable("w1", shape=[784,256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", shape=[256], initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("w2", shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", shape=[256], initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.sigmoid(tf.matmul(L1, W2) + b1)

W3 = tf.get_variable("w3", shape=[256,10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("b3", shape=[10], initializer=tf.contrib.layers.xavier_initializer())
logits = tf.matmul(L2, W3) + b3
hypothesis = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
opt = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

batch_size = 100

ckpt_path = "./model/checkpoint.ckpt"

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, ckpt_path)
    '''
    for epoch in range(15):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, opt], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c / total_batch
        print('Epoch:', '%d' % (epoch+1), 'cost =', '{:.9f}'.format(avg_cost))
    '''
    is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print("Gradient Descent Optimizer without dropout Accuracy", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

    saver.save(sess, ckpt_path)

