
# TfLearn version of DeepMNIST
# taking from https://www.tensorflow.org/get_started/mnist/pros
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Create input object which reads data from MNIST datasets.  
# Perform one-hot encoding to define the digit
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Define data shape
image_width = 28
image_height = 28
number_of_classes = 10

# Create a TensorFlow session
sess = tf.InteractiveSession()

# Define placeholders for MNIST input data
x = tf.placeholder(tf.float32, shape=[None, image_width * image_height])
y_ = tf.placeholder(tf.float32, [None, number_of_classes])  


# We now define the weights W and biases b for our model. 
W = tf.Variable(tf.zeros([image_width * image_height, number_of_classes]))
b = tf.Variable(tf.zeros([number_of_classes]))

# Before Variables can be used within a session, they must be initialized using that session
sess.run(tf.global_variables_initializer())

# regression model
y = tf.nn.softmax(tf.matmul(x,W)+b)

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Set training
# TensorFlow has a variety of built-in optimization algorithms.
# For this example, we will use steepest gradient descent,
# with a step length of 0.5, to descend the cross entropy.
learning_rate = 0.5
number_of_steps = 10000
batch_size = 100
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Train the model
for _ in range(number_of_steps):
    batch = mnist.train.next_batch(batch_size)  
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
	
# Evaluate the model 
correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Print out the accuracy
acc_eval = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print(f"Current accuracy: %{acc_eval * 100}")