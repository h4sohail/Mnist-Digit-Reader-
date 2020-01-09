
#   TfLearn version of DeepMNIST
# taking from https://www.tensorflow.org/get_started/mnist/pros
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Create input object which reads data from MNIST datasets.  
# Perform one-hot encoding to define the digit
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

# Define data shape
image_width = 28
image_height = 28
number_of_classes = 10
hidden_layer_size = 16

# Define placeholders for MNIST input data
x = tf.placeholder(tf.float32, shape=[None, image_width * image_height])
y_ = tf.placeholder(tf.float32, [None, number_of_classes])  

#To make a hidden layer, its input size MUST be the same as the previous layers output size
#Wn = tf.Variable(tf.zeros([input_size, output_size]))
#bn = tf.Variable(tf.zeros([output_size]))
#We now define the weights W1 and biases b1 for our first hidden layer 
W1 = tf.Variable(tf.zeros([image_width * image_height, hidden_layer_size]))
b1 = tf.Variable(tf.zeros([hidden_layer_size]))

#We now define the weights W2 and biases b2 for our second/last hidden layer 
W2 = tf.Variable(tf.zeros([hidden_layer_size, number_of_classes]))
b2 = tf.Variable(tf.zeros([number_of_classes]))

#Before Variables can be used within a session, they must be initialized using that session
sess.run(tf.global_variables_initializer())

# Neural network
z1 = tf.matmul(x, W1) + b1 #First set of logits
h1 = tf.nn.sigmoid(z1) #Apply activation function

#The last output layer is special for tf
z2 = tf.matmul(h1, W2) + b2 #Second set of logits
y = tf.nn.softmax(z2) #Apply activation function

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# Set training
#TensorFlow has a variety of built-in optimization algorithms.
# For this example, we will use steepest gradient descent,
# with a step length of 0.5, to descend the cross entropy.
learning_rate = 0.5
number_of_steps = 10000
batch_size = 100
savePath = 'tmp/tensor_model'
saver = tf.train.Saver() # This saves the session for later use 
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

# Save the path to the trained model
saver.save(sess, savePath)
print('Session saved in path '+savePath)