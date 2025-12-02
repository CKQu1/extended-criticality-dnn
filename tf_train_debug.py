import numpy as np
import os
import random
from os.path import join, isdir, isfile
from os import makedirs

#C_SIZE = 32 # channel size.
C_SIZE = 100
K_SIZE = 3 # kernel size
#LEARNING_RATE = 2e-2
#LEARNING_RATE = 3e-2
#LEARNING_RATE = 1.5e-2
LEARNING_RATE = 1e-2
#LEARNING_RATE = 1e-3
#LEARNING_RATE = 8e-3
#MOMENTUM = 0.95
MOMENTUM = 0

#BATCH_SIZE = 300
BATCH_SIZE = 1024 

# def run_model(alpha100, g100, seed,
#               depth, 
#               epochs, root_path):
    #global df, accuracy_log, loss_log

alpha100, g100, seed = 100, 100, 0
depth = 3
epochs = 10
root_path = '.droot/cnn_test'

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical    
from tqdm import tqdm

from tf_models import ConvModel_cpu, ConvModel_gpu

print('Settings')
print(f'C_SIZE = {C_SIZE}, K_SIZE = {K_SIZE}, lr = {LEARNING_RATE}, mom = {MOMENTUM}, bs = {BATCH_SIZE} \n')
print(f'alpha100 = {alpha100}, g100 = {g100}, seed = {seed}, depth = {depth}, epochs = {epochs} \n')

alpha100, g100, seed, depth, epochs = int(alpha100), int(g100), int(seed), int(depth), int(epochs)    

# SET SEED
tf.config.experimental.enable_op_determinism()
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000, seed=seed).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

# Ensure the model is leveraging GPU
device = '/GPU:0' if tf.config.experimental.list_physical_devices('GPU') else '/CPU:0'
print("Running on GPU" if tf.config.experimental.list_physical_devices('GPU') else "Running on CPU")
print('\n')

# Instantiate and train model
if 'GPU' in device:
    model = ConvModel_gpu(alpha=alpha100/100, g=g100/100, seed=seed, depth=depth,
                        c_size=C_SIZE, k_size=K_SIZE)
else:
    model = ConvModel_cpu(alpha=alpha100/100, g=g100/100, seed=seed, depth=depth,
                        c_size=C_SIZE, k_size=K_SIZE)

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

# Training function
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Evaluation function
def evaluate(dataset, model, loss_fn):
    avg_loss = tf.keras.metrics.Mean()
    accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    for images, labels in dataset:
        predictions = model(images, training=False)
        loss = loss_fn(labels, predictions)
        avg_loss.update_state(loss)
        accuracy_metric.update_state(labels, predictions)

    return avg_loss.result().numpy(), accuracy_metric.result().numpy()

# Training loop with device placement
metric_cols = ['train loss', 'train acc', 'test loss', 'test acc']
metrics_ls = []
save_dir = join(root_path, f'cnn{depth}_{alpha100}_{g100}_{seed}')
if not isdir(save_dir): makedirs(save_dir)    
for epoch in tqdm(range(epochs)):
    for images, labels in train_dataset:
        #with tf.device('/GPU:0'):  # Use '/CPU:0' if you want to force CPU for testing
        with tf.device(device):
            # loss = train_step(images, labels)
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_fn(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    

    # Evaluate on training and test sets
    with tf.device(device):
        train_loss, train_accuracy = evaluate(train_dataset, model, loss_fn)
        test_loss, test_accuracy = evaluate(test_dataset, model, loss_fn)

    metrics_ls.append([train_loss, train_accuracy, test_loss, test_accuracy])
    df = pd.DataFrame(metrics_ls, columns=metric_cols)
    df.to_csv(join(save_dir, '_acc_loss'))            

    print(f"Epoch {epoch + 1}:")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# save train acc/loss

if isfile(join(save_dir, '_acc_loss')):
    os.remove(join(save_dir, '_acc_loss'))            
df_path = join(save_dir, 'acc_loss')
df.to_csv(df_path)
print(f'Data saved as {df_path}')