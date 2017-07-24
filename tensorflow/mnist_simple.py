'''
MNIST implementation using tensorflow
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist_dataset = input_data.read_data_sets('../datasets/MNIST_data/', one_hot=True)

'''
    Neural network model
'''
# Images are 28x28, represented by a vector with 784 positions
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#weighted sum
z = tf.matmul(x, W) + b
#Activation
y = tf.nn.softmax(z)

'''
    Training with cross entropy loss function
'''
#Expected y
y_ = tf.placeholder(tf.float32, [None, 10])
#Loss function
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#train step
learning_rate = 0.05
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

'''
    Begin training execution of the defined model above
'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

epoch = 10000
print("begin train")

for i in range(epoch):
    #print("Running epoch %s" % (i))
    batch_xs, batch_ys = mnist_dataset.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

'''
    Evaluating...
'''
error = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(error, tf.float32))
'''
    feed dict is needed for tensorflow using as the placeholders values, 
    in this case, x and y_ are the only placeholders in this model
'''
print(sess.run(accuracy, feed_dict={x: mnist_dataset.test.images, y_: mnist_dataset.test.labels}))
