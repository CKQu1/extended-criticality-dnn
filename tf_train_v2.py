import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Set random seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000, seed=SEED).batch(300)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(300)

# Model definition as a subclass of tf.keras.Model
class ConvModel(tf.keras.Model):
    def __init__(self, alpha, g, depth):
        super(ConvModel, self).__init__()
        self.alpha = alpha
        self.g = g
        self.depth = depth
        self.phi = tf.tanh
        self.conv_layers = []
        
        # Define convolutional layers
        self.conv_layers.append(self.get_weight_layer([3, 3, 1, 32], 'kernel_0'))
        for _ in range(2):
            self.conv_layers.append(self.get_weight_layer([3, 3, 32, 32], 'reduction_kernel'))
        
        for _ in range(depth):
            self.conv_layers.append(self.get_weight_layer([3, 3, 32, 32], 'block_conv'))
        
        # Final dense layer
        self.logit_W = self.add_weight(shape=[32, 10], initializer='random_uniform', name='logit_W')

    def get_weight_layer(self, shape, name):
        return tf.Variable(tf.random.normal(shape=shape, mean=0.0, stddev=0.1, seed=SEED), name=name)
    
    def call(self, inputs, training=False):
        z = tf.reshape(inputs, [-1, 28, 28, 1])
        for layer in self.conv_layers:
            z = tf.nn.conv2d(z, layer, strides=[1, 1, 1, 1], padding='SAME')
            z = self.phi(z)
        z_ave = tf.reduce_mean(z, axis=[1, 2])
        return tf.matmul(z_ave, self.logit_W)

# Instantiate and train model
model = ConvModel(alpha=1.5, g=0.5, depth=5)

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-3)

# Training function
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(10):
    for images, labels in train_dataset:
        loss = train_step(images, labels)
    print(f'Epoch {epoch}, Loss: {loss.numpy()}')
