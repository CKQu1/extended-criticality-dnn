import numpy as np
import tensorflow as tf

def get_weight(shape, alpha, g, seed, name=None):    
    from scipy.stats import levy_stable
    # most correct
    #N_eff = int(shape[0]*shape[1]) * shape[0]  # incorrect    
    N_eff = int(shape[0]*shape[1]) * shape[2]  # debugged on 2024/03/27           
    return tf.Variable(levy_stable.rvs(alpha, 0, size=shape, scale=g*(0.5/N_eff)**(1./alpha), random_state=seed), # set seed
                        name=name, dtype=tf.float32)

def get_uniform_weight(shape):
    N_in = shape[0]  # checked on 2024/03/27 
    initializer = tf.random_uniform_initializer(minval=-1/N_in**0.5, maxval=1/N_in**0.5)
    return tf.Variable(initializer(shape=shape, dtype=tf.float32))                            

def conv2d(x, w, strides=1, padding='SAME'):
    return tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)        

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

# Model definition as a subclass of tf.keras.Model
class ConvModel_cpu(tf.keras.Model):
    def __init__(self, alpha, g, seed, depth,
                 c_size, k_size):
        super(ConvModel_cpu, self).__init__()
        self.alpha = alpha
        self.g = g
        self.seed = seed
        self.depth = depth        
        self.c_size = c_size
        self.k_size = k_size

        self.phi = tf.tanh
        self.conv_layers = []
        
        np.random.seed(self.seed)
        # self.levy_stable_seeds = np.random.randint(1000000, size=depth+3)
        self.levy_stable_seeds = np.random.randint(1000000, size=depth)

        # Define convolutional layers
        self.conv_layers.append(get_weight([k_size, k_size, 1, c_size], alpha, g, self.levy_stable_seeds[0], name='kernel_0'))
        shape = [k_size, k_size, c_size, c_size]
        # for j in range(2):
        #     self.conv_layers.append(get_weight(shape, alpha, g, self.levy_stable_seeds[j+1], name='reduction_{}_kernel'.format(j)))        
        # for j in range(depth):
        #     name = 'block_conv_{}'.format(j)
        #     kernel_name, bias_name = name + 'kernel', name + 'bias'
        #     #kernel = get_orthogonal_weight(kernel_name, shape, gain=std)
        #     self.conv_layers.append(get_weight(shape, alpha, g, self.levy_stable_seeds[j+3], name='reduction_{}_kernel'.format(j)))
              
        for j in range(depth-1):
            name = 'block_conv_{}'.format(j+1)
            kernel_name, bias_name = name + 'kernel', name + 'bias'
            #kernel = get_orthogonal_weight(kernel_name, shape, gain=std)
            self.conv_layers.append(get_weight(shape, alpha, g, self.levy_stable_seeds[j+1], name='reduction_{}_kernel'.format(j+1)))

        # Final dense layer
        #self.logit_W = self.add_weight(shape=[c_size, 10], initializer='random_uniform', name='logit_W')
        self.logit_W = get_uniform_weight([c_size, 10])
    
    def call(self, inputs, training=False):
        z = tf.reshape(inputs, [-1, 28, 28, 1])
        #for layer in self.conv_layers:
        new_width = 7 # width of the current image after dimension reduction. 
        for lidx, layer in enumerate(self.conv_layers):
            #z = tf.nn.conv2d(z, layer, strides=[1, 1, 1, 1], padding='SAME')
            if lidx == 0:
                z = conv2d(z, layer, strides=1, padding='SAME')
            elif lidx in [1,2]:
                z = conv2d(z, layer, strides=2)
            else:
                z_pad = circular_padding(z, new_width, self.k_size)
                z = conv2d(z_pad, layer, padding='VALID')

            z = self.phi(z)
        z_ave = tf.reduce_mean(z, axis=[1, 2])
        return tf.matmul(z_ave, self.logit_W)   
    

class ConvModel_gpu(tf.keras.Model):
    def __init__(self, alpha, g, seed, depth, c_size, k_size):
        super(ConvModel_gpu, self).__init__()
        self.alpha = alpha
        self.g = g
        self.seed = seed
        self.depth = depth        
        self.c_size = c_size
        self.k_size = k_size

        self.phi = tf.tanh
        self.conv_layers = []

        np.random.seed(self.seed)
        # self.levy_stable_seeds = np.random.randint(1000000, size=depth+3)
        self.levy_stable_seeds = np.random.randint(1000000, size=depth)

        # First conv layer
        init_val = get_weight([k_size, k_size, 1, c_size], alpha, g, self.levy_stable_seeds[0])
        self.conv_layers.append(
            self.add_weight(
                name='kernel_0',
                shape=init_val.shape,
                trainable=True,
                initializer=tf.constant_initializer(init_val.numpy())
            )
        )

        # Subsequent conv layers
        shape = [k_size, k_size, c_size, c_size]
        # for j in range(2 + depth):
        for j in range(depth-1):
            init_val = get_weight(shape, alpha, g, self.levy_stable_seeds[j+1])
            self.conv_layers.append(
                self.add_weight(
                    name=f'reduction_{j+1}_kernel',
                    shape=init_val.shape,
                    trainable=True,
                    initializer=tf.constant_initializer(init_val.numpy())
                )
            )

        # Final dense layer
        init_val = get_uniform_weight([c_size, 10])
        self.logit_W = self.add_weight(
            name='logit_W',
            shape=init_val.shape,
            trainable=True,
            initializer=tf.constant_initializer(init_val.numpy())
        )

    def call(self, inputs, training=False):
        z = tf.reshape(inputs, [-1, 28, 28, 1])
        new_width = 7  # depends on your architecture
        for lidx, layer in enumerate(self.conv_layers):
            if lidx == 0:
                z = conv2d(z, layer, strides=1, padding='SAME')
            elif lidx in [1,2]:
                z = conv2d(z, layer, strides=2)
            else:
                z_pad = circular_padding(z, new_width, self.k_size)
                z = conv2d(z_pad, layer, padding='VALID')
            z = self.phi(z)
        z_ave = tf.reduce_mean(z, axis=[1,2])
        return tf.matmul(z_ave, self.logit_W)
