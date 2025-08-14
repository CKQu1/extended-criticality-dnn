"""
From the notebook https://github.com/brain-research/mean-field-cnns/blob/master/Delta_Orthogonal_Convolution_Demo.ipynb
based on https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/autoencoder.ipynb
"""
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import levy_stable

from tqdm import tqdm
from os import makedirs
from os.path import isdir, isfile, join

"""
# variances of weight and bias. 
# To obtain critical values of the variances of the weights and biases,
# see compute mean field below.  
mf = MeanField(np.tanh, d_tanh)
qstar = 1./DEPTH
W_VAR, B_VAR = mf.sw_sb(qstar, 1)
"""

C_SIZE = 32 # channel size.
#C_SIZE = 100
K_SIZE = 3 # kernel size
#LEARNING_RATE = 1e-2
#LEARNING_RATE = 1e-3
LEARNING_RATE = 5e-3
#MOMENTUM = 0.95
MOMENTUM = 0

BATCH_SIZE = 300

def circular_padding(input_, width, kernel_size):
        
    """Padding input_ for computing circular convolution."""
    begin = kernel_size // 2
    end = kernel_size - 1 - begin
    tmp_up = tf.slice(input_, [0, width - begin, 0, 0], [-1, begin, width, -1])
    tmp_down = tf.slice(input_, [0, 0, 0, 0], [-1, end, width, -1])
    tmp = tf.concat([tmp_up, input_, tmp_down], 1)
    new_width = width + kernel_size - 1
    tmp_left = tf.slice(tmp, [0, 0, width - begin, 0], [-1, new_width, begin, -1])
    tmp_right = tf.slice(tmp, [0, 0, 0, 0], [-1, new_width, end, -1])
    return  tf.concat([tmp_left, tmp, tmp_right], 2)

def conv2d(x, w, strides=1, padding='SAME'):
    import tensorflow as tf
    return tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)

def get_weight(shape, alpha, g, seed, name=None):    
    import tensorflow as tf
    # most correct
    #N_eff = int(shape[0]*shape[1]) * shape[0]  # incorrect    
    N_eff = int(shape[0]*shape[1]) * shape[2]  # debugged on 2024/03/27   
    np.random.seed(seed)                             
    return tf.Variable(levy_stable.rvs(alpha, 0, size=shape, scale=g*(0.5/N_eff)**(1./alpha)), 
                       name=name, dtype=tf.float32)

def get_uniform_weight(shape):
    import tensorflow as tf
    N_in = shape[0]  # checked on 2024/03/27 
    initializer = tf.random_uniform_initializer(minval=-1/N_in**0.5, maxval=1/N_in**0.5)
    return tf.Variable(initializer(shape=shape, dtype=tf.float32)) 
                                
def get_orthogonal_weight(name, shape, gain=1.):
    import tensorflow as tf
    # Can also use tf.contrib.framework.convolutional_orthogonal_2d
    return tf.get_variable(name, shape=shape,
        initializer=tf.contrib.framework.convolutional_delta_orthogonal(gain=gain))

def conv_model(alpha, g, depth, x):
    import tensorflow as tf
    phi = tf.tanh # non-linearity
    """Convolutional layers. Ouput logits. """
    z = tf.reshape(x, [-1,28,28,1])

    levy_stable_seeds = np.random.randint(1000000, size=depth+3)
    # Increase the channel size to C_SIZE.
    kernel = get_weight([K_SIZE, K_SIZE, 1, C_SIZE], alpha, g, levy_stable_seeds[0], name='kernel_0')
    h = conv2d(z, kernel, strides=1, padding='SAME')
    z = phi(h)

    # Reducing spacial dimension to 7 * 7; applying conv with stride=2 twice.
    shape = [K_SIZE, K_SIZE, C_SIZE, C_SIZE]
    for j in range(2):
        kernel = get_weight(shape, alpha, g, levy_stable_seeds[j+1], name='reduction_{}_kernel'.format(j))
        h = conv2d(z, kernel, strides=2)
        z = phi(h)
    new_width = 7 # width of the current image after dimension reduction. 

    # A deep convolution block with depth=depth.
    """
    std = np.sqrt(W_VAR)
    for j in range(depth):
        name = 'block_conv_{}'.format(j)
        kernel_name, bias_name = name + 'kernel', name + 'bias'
        kernel = get_orthogonal_weight(kernel_name, shape, gain=std)
        bias = get_weight([C_SIZE], std=np.sqrt(B_VAR), name=bias_name)
        z_pad = circular_padding(z, new_width, K_SIZE)
        h = conv2d(z_pad, kernel, padding='VALID') + bias
        z = phi(h)
    """
    for j in range(depth):
        name = 'block_conv_{}'.format(j)
        kernel_name, bias_name = name + 'kernel', name + 'bias'
        #kernel = get_orthogonal_weight(kernel_name, shape, gain=std)
        kernel = get_weight(shape, alpha, g, levy_stable_seeds[j+3], name='reduction_{}_kernel'.format(j))
        z_pad = circular_padding(z, new_width, K_SIZE)
        h = conv2d(z_pad, kernel, padding='VALID')
        z = phi(h)    

    z_ave = tf.reduce_mean(z, [1, 2])
    #logit_W = get_weight([C_SIZE, 10], std=np.sqrt(1./(C_SIZE)))
    logit_W = get_uniform_weight([C_SIZE, 10])
    #logit_b = get_weight([10], std=0.)
    #return tf.matmul(z_ave, logit_W) + logit_b
    return tf.matmul(z_ave, logit_W)

def loss(logits, labels):
    import tensorflow as tf
    # return tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))    

def train_op(loss_, learning_rate, global_step):
    import tensorflow as tf
    #with tf.control_dependencies([tf.assign(global_step, global_step + 1)]):
    with tf.control_dependencies([tf.compat.v1.assign(global_step, global_step + 1)]):
        #return tf.train.MomentumOptimizer(learning_rate, MOMENTUM).minimize(loss_)
        return tf.compat.v1.train.MomentumOptimizer(learning_rate, MOMENTUM).minimize(loss_)

def accuracy(logits, labels):
    import tensorflow as tf
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1),
                                            tf.argmax(labels, 1)), 
                                tf.float32))

#def run_model(alpha, g, num_steps=1000):    
def run_model(alpha100, g100, seed,
              depth, 
              epochs, root_path):
    #global df, accuracy_log, loss_log

    import matplotlib.pylab as plt
    import numpy as np
    import tensorflow as tf
    import pandas as pd
    import warnings
    from sklearn import preprocessing

    print('Settings')
    print(f'C_SIZE = {C_SIZE}, K_SIZE = {K_SIZE}, lr = {LEARNING_RATE}, mom = {MOMENTUM}, bs = {BATCH_SIZE} \n')

    # set seed
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

    # load data
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #mnist, mnist_test = tf.keras.datasets.mnist.load_data()
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #join('.keras', 'datasets', 'mnist.npz')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('/home/chqu7424/.keras/datasets/mnist.npz')
    x_train = x_train.reshape(-1,784)
    x_test = x_test.reshape(-1,784)
    # one hot encoding
    # based on https://github.com/christianversloot/machine-learning-articles/blob/main/one-hot-encoding-for-machine-learning-with-python-and-scikit-learn.md
    ohe = preprocessing.OneHotEncoder()
    # train
    y_train = y_train.reshape(-1, 1)
    ohe.fit(y_train)
    y_train = ohe.transform(y_train).toarray()
    # test
    y_test = y_test.reshape(-1, 1)
    ohe.fit(y_test)
    y_test = ohe.transform(y_test).toarray()    

    tf.compat.v1.disable_eager_execution()

    # initialization
    alpha100, g100 = int(alpha100), int(g100)
    epochs = int(epochs)
    alpha, g = alpha100/100, g100/100
    if not isdir(root_path): makedirs(root_path)

    # CNN setting
    # depth and C_SIZE are chosen to be small for a fast running demo. You will want
    # to increase both values for most use cases.
    # Further note: this will run *much faster* if you choose a runtime with a GPU
    # accelerator.
    #depth = 16 # number of layers.
    depth = int(depth)

    #tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()    
    accuracy_log, loss_log, test_accuracy_log, test_loss_log = [], [], [], []
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
    #global_step = tf.train.get_or_create_global_step()
    global_step = tf.compat.v1.train.get_or_create_global_step()
    logits = conv_model(alpha, g, depth, x)
    acc, loss_ = accuracy(logits, y_), loss(logits, y_)
    training = train_op(loss_, LEARNING_RATE, global_step)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        total_steps = len(x_train)//BATCH_SIZE + int(len(x_train)%BATCH_SIZE!=0)   
        shuffled_indices = np.arange(len(x_train))      
        np.random.seed(seed)
        for epoch in tqdm(range(1,epochs+1)):
            # reshuffle
            np.random.shuffle(shuffled_indices)
            images = x_train[shuffled_indices]
            labels = y_train[shuffled_indices]
            for i in range(total_steps):
            #for i in range(num_steps):
            #for i in tqdm(range(num_steps)):
                j = i % total_steps
                b1 = j*BATCH_SIZE
                b2 = min((j+1)*BATCH_SIZE, len(x_train))             
                batch_xs = images[b1:b2]
                batch_ys = labels[b1:b2]
                #print(f'batch_xs shape: {batch_xs.shape}')  # delete
                #print(f'batch_ys shape: {batch_ys.shape}')  # delete

                _, acc_value, loss_value, g_step = sess.run(
                    [training, acc, loss_, global_step], 
                    feed_dict={x:batch_xs, y_:batch_ys})
                # accuracy_log.append(acc_value)
                # loss_log.append(loss_value)
                # if i % (num_steps/20) == 0 or i == num_steps-1:
                #     print('Step: %5d Accuracy: %.2f Loss: %g'%(g_step, acc_value, loss_value))
            accuracy_log.append(acc_value)
            loss_log.append(loss_value)        

            # Evaluation
            test_acc_value, test_loss_value = sess.run(
                [acc, loss_],
                feed_dict={x: x_test, y_: y_test}
            )

            test_accuracy_log.append(test_acc_value)
            test_loss_log.append(test_loss_value)  

            print('Epoch: %5d Train Accuracy: %.2f Train Loss: %g Train Accuracy: %.2f Train Loss: %g'
            %(epoch, acc_value, loss_value, test_acc_value, test_loss_value))

            # old
            #print('Epoch: %5d Train Accuracy: %.2f Train Loss %g :'%(epoch, acc_value, loss_value))            

    # save train acc/loss
    df = pd.DataFrame(columns=['train loss', 'train acc', 'test loss', 'test acc'], dtype=object)
    df.loc[:,'train loss'] = loss_log
    df.loc[:,'train acc'] = accuracy_log  
    df.loc[:,'test loss'] = test_loss_log
    df.loc[:,'test acc'] = test_accuracy_log        
    df_path = join(root_path, 'acc_loss')
    df.to_csv(df_path)
    print(f'Data saved as {df_path}')

    #return accuracy_log, loss_log

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
    pbs_array_data = [(alpha100, g100, SEED, DEPTH, epochs, join(models_path,f'cnn{DEPTH}_{alpha100}_{g100}'))
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
    net_type = "fc"
    
    train_seeds = [0]
    alpha100s = list(range(100,201,5))  # full grid 2
    g100s = list(range(20,301,20))  # full grid 2    
    # alpha100s = list(range(100,201,10))  # full grid 1
    # g100s = list(range(25, 301, 25))  # full grid 1

    n_subjobs = 315
    for train_seed in train_seeds:

        triplets = [] 
        for train_seed in train_seeds:
            for alpha100 in alpha100s:
                for g100 in g100s:
                    triplets.append(f'{train_seed}_{alpha100}_{g100}')
        
        chunks = int(np.ceil(len(triplets)/n_subjobs))
        #chunks = 11
        tripletss = list_str_divider(triplets, chunks)

        epochs = 100

        SEED = train_seed
        DEPTH = 6
        net_type = 'cnn_cpad'  # cnn with circular pad
        fc_init = "fc_default"
        dataset = 'mnist'
        optimizer = 'sgd'

        models_path = join(DROOT, 'trained_cnns', 
                           f'cnn{DEPTH}_{fc_init}_{dataset}_{optimizer}_epochs={epochs}_seed={SEED}')
        if not isdir(models_path): makedirs(models_path)

        # raw submittions    
        pbs_array_data = [(','.join(literal_eval(triplets)),
                            DEPTH, epochs, join(models_path,f'cnn{DEPTH}_{alpha100}_{g100}')
                            )
                            for triplets in tripletss
                            ]   

        # save training settings
        df_settings = pd.DataFrame(columns=['c_size', 'k_size', 'lr', 'momentum', 'batch_size'])
        df_settings.loc[0,:] = [C_SIZE, K_SIZE, LEARNING_RATE, MOMENTUM, BATCH_SIZE]
        df_settings.to_csv(join(models_path, 'settings.csv'))

        ncpus, ngpus = 1, 0
        command = command_setup(SPATH, bind_path=BPATH, ncpus=ncpus, ngpus=ngpus)   

        perm, pbss = job_divider(pbs_array_data, len(project_ls))
        quit()  # delete
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


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])