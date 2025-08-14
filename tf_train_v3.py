import numpy as np
import os
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

def run_model(alpha100, g100, seed,
              depth, 
              epochs, root_path):
    #global df, accuracy_log, loss_log

    import numpy as np
    import tensorflow as tf
    import pandas as pd
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical    
    from tqdm import tqdm

    from tf_models import ConvModel

    print('Settings')
    print(f'C_SIZE = {C_SIZE}, K_SIZE = {K_SIZE}, lr = {LEARNING_RATE}, mom = {MOMENTUM}, bs = {BATCH_SIZE} \n')
    print(f'alpha100 = {alpha100}, g100 = {g100}, seed = {seed}, depth = {depth}, epochs = {epochs} \n')

    alpha100, g100, seed, depth, epochs = int(alpha100), int(g100), int(seed), int(depth), int(epochs)    

    # SET SEED
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

    # Instantiate and train model
    model = ConvModel(alpha=alpha100/100, g=g100/100, seed=seed, depth=depth,
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

    # Ensure the model is leveraging GPU
    device = '/GPU:0' if tf.config.experimental.list_physical_devices('GPU') else '/CPU:0'
    print("Running on GPU" if tf.config.experimental.list_physical_devices('GPU') else "Running on CPU")
    print('\n')

    # Training loop with device placement
    metric_cols = ['train loss', 'train acc', 'test loss', 'test acc']
    metrics_ls = []
    save_dir = join(root_path, f'cnn{depth}_{alpha100}_{g100}_{seed}')
    if not isdir(save_dir): makedirs(save_dir)    
    for epoch in tqdm(range(epochs)):
        for images, labels in train_dataset:
            #with tf.device('/GPU:0'):  # Use '/CPU:0' if you want to force CPU for testing
            with tf.device(device):
                loss = train_step(images, labels)

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

# func: run_model
def submit(*args):    
    #global command, pbs_array_data, df_settings

    import pandas as pd
    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH, DROOT

    alpha100s = list(range(100,201,10))
    g100s = list(range(25, 301, 25))
    # alpha100s = [100,200]
    # g100s = [25,100,300]
    epochs = 50

    SEED = 0
    DEPTH = 4
    net_type = 'cnn_cpad'  # cnn with circular pad
    fc_init = "fc_default"
    dataset = 'mnist'
    optimizer = 'sgd'

    models_path = join(DROOT, 'trained_cnns', f'cnn{DEPTH}_{fc_init}_{dataset}_{optimizer}_epochs={epochs}')
    if not isdir(models_path): makedirs(models_path)
    pbs_array_data = [(alpha100, g100, SEED, DEPTH, epochs, join(models_path,f'cnn{DEPTH}_{SEED}'))
                      for alpha100 in alpha100s
                      for g100 in g100s
                      ]

    # save training settings
    df_settings = pd.DataFrame(columns=['c_size', 'k_size', 'lr', 'momentum', 'batch_size'])
    df_settings.loc[0,:] = [C_SIZE, K_SIZE, LEARNING_RATE, MOMENTUM, BATCH_SIZE]
    df_settings.to_csv(join(models_path, 'settings.csv'))

    ncpus, ngpus = 1, 0
    command = command_setup(singularity_path, bind_path=bind_path, ncpus=ncpus, ngpus=ngpus)   

    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    #quit()  # delete
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=models_path,
             P=project_ls[pidx],
             ngpus=ngpus,
             ncpus=ncpus,
             walltime='23:59:59',
             #walltime='23:59:59',
             mem='5GB')

def batch_run_model(triplets, depth, epochs, root_path):
    triplets = triplets.split(',')
    for triplet in triplets:
        alpha100, g100, train_seed = triplet.split('_')
        alpha100, g100, train_seed = int(alpha100), int(g100), int(train_seed)
        run_model(alpha100, g100, train_seed,
                  depth, 
                  epochs, root_path)

# func: batch_run_model
def batch_submit(*args):
    import pandas as pd
    from qsub import qsub, job_divider, project_ls, command_setup
    from constants import SPATH, BPATH, DROOT    
    from ast import literal_eval

    """
    Batch training mnist with MLPs
    """
    global pbs_array_data, tripletss

    from qsub import qsub, job_divider, project_ls, command_setup, list_str_divider
    from constants import SPATH, BPATH

    dataset_name = "mnist"
    
    train_seeds = [0]
    alpha100s = list(range(100,201,5))  # full grid 2
    g100s = list(range(20,301,20))  # full grid 2    
    # alpha100s = list(range(100,201,10))  # full grid 1
    # g100s = list(range(25, 301, 25))  # full grid 1
    # alpha100s = [100,150,200]
    # g100s = [25,100,300]

    n_subjobs = 6  # GPUs
    #n_subjobs = 132
    #n_subjobs = 315
    for train_seed in train_seeds:

        triplets = [] 
        for train_seed in train_seeds:
            for alpha100 in alpha100s:
                for g100 in g100s:
                    triplets.append(f'{alpha100}_{g100}_{train_seed}')
        
        chunks = int(np.ceil(len(triplets)/n_subjobs))
        #chunks = 11
        tripletss = list_str_divider(triplets, chunks)

        epochs = 100

        SEED = train_seed
        DEPTH = 7
        net_type = 'cnn_cpad'  # cnn with circular pad
        fc_init = "fc_default"
        dataset = 'mnist'
        optimizer = 'sgd'

        # models_path = join(DROOT, 'trained-cnns-v5', 
        #                    f'cnn{DEPTH}_seed={SEED}')
        models_path = join(DROOT, 'wide-cnns', 
                           f'cnn{DEPTH}_seed={SEED}')        
        if not isdir(models_path): makedirs(models_path)

        # raw submittions    
        pbs_array_data = [(','.join(literal_eval(triplets)),
                            DEPTH, epochs, models_path
                            )
                            for triplets in tripletss
                            ]   

        # save training settings
        df_cols = ['net_type', 'fc_init', 'dataset', 'optimizer',
                   'depth', 'c_size', 'k_size', 'lr', 'momentum', 'batch_size']
        df_settings = pd.DataFrame(columns=df_cols)
        df_settings.loc[0,:] = [net_type, fc_init, dataset, optimizer,
                                DEPTH, C_SIZE, K_SIZE, LEARNING_RATE, MOMENTUM, BATCH_SIZE]
        df_settings.to_csv(join(models_path, 'settings.csv'))

        #ncpus, ngpus = 1, 0
        ncpus, ngpus = 1, 1
        #command = command_setup(SPATH, bind_path=BPATH, ncpus=ncpus, ngpus=ngpus)  # use container   
        command = command_setup('', bind_path=BPATH, ncpus=ncpus, ngpus=ngpus)

        print(f'Total jobs: {len(pbs_array_data)} \n')

        perm, pbss = job_divider(pbs_array_data, len(project_ls))
        #quit()  # delete
        for idx, pidx in enumerate(perm):
            pbs_array_true = pbss[idx]
            print(project_ls[pidx])
            qsub(f'{command} {sys.argv[0]} {" ".join(args)}',    
                pbs_array_true, 
                path=models_path,
                P=project_ls[pidx],
                ngpus=ngpus,
                ncpus=ncpus,
                # use source
                add_command='module purge; module load python/3.7.7 cuda/10.2.89 openmpi-gcc/4.1.1',
                source='tf_cuda/bin/activate',  
                walltime='23:59:59',
                mem='6GB')

# ----------------------------------------
 
# # Set random seed for reproducibility
# SEED = 0
# tf.random.set_seed(SEED)
# np.random.seed(SEED)

# # Load and preprocess MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000, seed=SEED).batch(BATCH_SIZE)
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

# # Instantiate and train model
# DEPTH = 5
# model = ConvModel(alpha=1.5, g=0.5, seed=SEED, depth=DEPTH)

# loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

# # Training function
# @tf.function
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         predictions = model(images, training=True)
#         loss = loss_fn(labels, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss

# # Evaluation function
# def evaluate(dataset, model, loss_fn):
#     avg_loss = tf.keras.metrics.Mean()
#     accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

#     for images, labels in dataset:
#         predictions = model(images, training=False)
#         loss = loss_fn(labels, predictions)
#         avg_loss.update_state(loss)
#         accuracy_metric.update_state(labels, predictions)

#     return avg_loss.result().numpy(), accuracy_metric.result().numpy()

# # Training loop
# EPOCHS = 3
# # for epoch in range(EPOCHS):
# #     for images, labels in train_dataset:
# #         loss = train_step(images, labels)
# #     print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# #     # Evaluate on training and test sets
# #     train_loss, train_accuracy = evaluate(train_dataset, model, loss_fn)
# #     test_loss, test_accuracy = evaluate(test_dataset, model, loss_fn)

# #     print(f"Epoch {epoch + 1}:")
# #     print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
# #     print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")    

# # Ensure the model is leveraging GPU
# device = '/GPU:0' if tf.config.experimental.list_physical_devices('GPU') else '/CPU:0'
# print("Running on GPU" if tf.config.experimental.list_physical_devices('GPU') else "Running on CPU")

# # Training loop with device placement
# for epoch in tqdm(range(10)):
#     for images, labels in train_dataset:
#         #with tf.device('/GPU:0'):  # Use '/CPU:0' if you want to force CPU for testing
#         with tf.device(device):
#             loss = train_step(images, labels)

#     # Evaluate on training and test sets
#     with tf.device(device):
#         train_loss, train_accuracy = evaluate(train_dataset, model, loss_fn)
#         test_loss, test_accuracy = evaluate(test_dataset, model, loss_fn)

#     print(f"Epoch {epoch + 1}:")
#     print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
#     print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
# ----------------------------------------        

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])