
import tensorflow as tf
import numpy as np


# Network Parameters
n_input = 625  
#n_input = 163840 # data input 
n_classes = 8 # fMRI total number of classes
dropout = 0.9 # Dropout, probability to keep units
learning_rate = 0.001
training_iters = 10000000
batch_size = 100
display_step = 10



print('loading files...')
f = file("../data/subj1/data_masked.bin","rb")
x_train = np.load(f)
y_train = np.load(f).astype(int)
# load the test data
x_test = np.load(f)
y_test = np.load(f).astype(int)
f.close()

# getting the data into shape
x_train = np.append(np.ones((x_train.shape[0], n_input - x_train.shape[1])), x_train, axis=1)
x_test = np.append(np.ones((x_test.shape[0], n_input - x_test.shape[1])), x_test, axis=1)




# returns the next mini-batch
def nextBatch(x, y):
    # Shuffle the data
    perm = np.arange(x.shape[0])
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    # select next batch
    return x[0:batch_size], y[0:batch_size]


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 25, 25, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()



logs = np.zeros((training_iters/(batch_size*display_step), 4))
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = nextBatch(x_train, y_train)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            # Calculate accuracy for the test data
            test_acc = sess.run(accuracy, feed_dict={x: x_test,
                          y: y_test,
                          keep_prob: 1.})

            # print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
            #       "{:.2f}".format(loss) + ", Training Accuracy= " + \
            #       "{:.2f}".format(acc*100) + ", Testing Accuracy:" + \
            #       "{:.2f}".format(test_acc*100))

            index = step/display_step - 1
            logs[index,0] = step*batch_size
            logs[index,1] = loss
            logs[index,2] = acc*100
            logs[index,3] = test_acc*100
        step += 1
    print("Optimization Finished!")



np.savetxt("./Logs/logs0.txt", logs, fmt='%.2f')


# for saving the graph to tensorboard   
summary_writer = tf.summary.FileWriter('./Logs', graph=sess.graph)
