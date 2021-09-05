#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


tf.set_random_seed(777)

sample = "if you want you"
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}

print(idx2char)
print(char2idx)


# In[3]:


dic_size = len(char2idx)
hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
sequence_length = len(sample)-1
learning_rate = 0.1


# In[4]:


sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]


# In[5]:


X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

x_one_hot = tf.one_hot(X, num_classes)

cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, dtype=tf.float32)


# In[6]:


X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)


# In[7]:


outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)


# In[8]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        
        result_str = [idx2char[c] for c in np.squeeze(result)]
        
        print(i, "loss:", l, "Prediction:", ''.join(result_str))

