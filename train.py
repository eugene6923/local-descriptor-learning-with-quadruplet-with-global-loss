import os
import os.path
from os import path

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2
from pylab import *
from sklearn.metrics import roc_curve,roc_auc_score
from datetime import datetime,timedelta
import wandb
from GenericTools import *
#from TripletLossTools import *
from onlyQuadru import *


from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform

from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import tensorflow as tf
mnist = tf.keras.datasets.mnist

# Allow memory growth for the GPU
#print(tf.test.is_gpu_available())

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

projectName = "27_mnist_10_mining"
project_path = './{0}/'.format(projectName)
model_path = './model/{0}/'.format(projectName)
if not path.exists(project_path):
    os.mkdir(project_path)
if not path.exists(model_path):
    os.mkdir(model_path)

nb_classes = 10
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

def buildDataSet():
    """Build dataset for train and test
    
    
    returns:
        dataset : list of lengh 10 containing images for each classes of shape (?,28,28,1)
    """
    (x_train_origin, y_train_origin), (x_test_origin, y_test_origin) = mnist.load_data()

    assert K.image_data_format() == 'channels_last'
    x_train_origin = x_train_origin.reshape(x_train_origin.shape[0], img_rows, img_cols, 1)
    x_test_origin = x_test_origin.reshape(x_test_origin.shape[0], img_rows, img_cols, 1)
    
    dataset_train = []
    dataset_test = []
    
    #Sorting images by classes and normalize values 0=>1
    for n in range(nb_classes):
        images_class_n = np.asarray([row for idx,row in enumerate(x_train_origin) if y_train_origin[idx]==n])
        dataset_train.append(images_class_n/255)
        
        images_class_n = np.asarray([row for idx,row in enumerate(x_test_origin) if y_test_origin[idx]==n])
        dataset_test.append(images_class_n/255)
        
    return dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin

dataset_train,dataset_test,dataset_train_flat_X,dataset_train_flat_Y,dataset_test_flat_X,dataset_test_flat_Y = buildDataSet()
print("Checking shapes for class 0 (train) : ",dataset_train[0].shape)
print("Checking shapes for class 0 (test) : ",dataset_test[0].shape)

def build_network(input_shape, embeddingsize):
    '''
    Define the neural network to learn image similarity
    Input : 
            input_shape : shape of input images
            embeddingsize : vectorsize used to encode our picture   
    '''

    # Convolutional Neural Network
    network = Sequential()
    network.add(Conv2D(128, (7,7), activation='relu',padding='same',
                     input_shape=input_shape,
                     kernel_initializer='he_uniform',
                     kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())
    network.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',padding='same',
                     kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())
    network.add(Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform',padding='same',
                     kernel_regularizer=l2(2e-4)))
    network.add(Flatten())
    network.add(Dense(4096, activation='relu',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform'))
    
    
    network.add(Dense(embeddingsize, activation=None,
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform'))
    
    #Force the encoding to live on the d-dimentional hypershpere
    network.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    
    return network


alpha1=1
alpha2=0.5
ramda=0.8
t1=0.1
t2=0.01
gamma=0.8
embeddingsize=10
nb_test_class=10

#network3 = build_network(input_shape,embeddingsize=embeddingsize)
#network3_train = build_model3(input_shape,network3,margin=alpha1)
#optimizer3 = Adam(lr = 0.00006)
#network3_train.compile(loss=None,optimizer=optimizer3)
#

def build_metric_network(single_embedding_shape):
    '''
    Define the neural network to learn the metric
    Input :
            single_embedding_shape : shape of input embeddings or feature map. Must be an array

    '''
    # compute shape for input
    input_shape = single_embedding_shape
    # the two input embeddings will be concatenated
    input_shape[0] = input_shape[0] * 2

    # Neural Network
    network = Sequential(name="learned_metric")
#    network.add(Conv2D(128, (7,7), activation='relu',padding='same',
#                     input_shape=input_shape,
#                     kernel_initializer='he_uniform',
#                     kernel_regularizer=l2(2e-4))
    network.add(Dense(10, activation='relu',
                      input_shape=input_shape,
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))
    network.add(Dense(10, activation='relu',
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer='he_uniform'))

    # Last layer : binary softmax
    network.add(Dense(2, activation='softmax'))

    # Select only one output value from the softmax
    network.add(Lambda(lambda x: x[:, 0]))

    return network

#modele en 4xloss
network4 = build_network(input_shape,embeddingsize=10)
#network4.set_weights(network3.get_weights()) #copy weights to have identical networks
metric_network4 = build_metric_network(single_embedding_shape=[10])
network4_train = build_model4(input_shape,network4,metric_network4,margin=alpha1, margin2=alpha2,gamma=gamma,ramda=ramda,t1=t1,t2=t2)
optimizer4 = Adam(lr = 0.00006)
network4_train.compile(loss=None,optimizer=optimizer4)
network4_train.summary()
#plot_model(network4_train,show_shapes=True, show_layer_names=True, to_file=project_path+'model_summary_4x.png')

n_iteration=0

quadruplets = get_batch_random(2,dataset_train)
print("Checking batch width, should be 4 : ",len(quadruplets))
print("Shapes in the batch A:{0} P:{1} N:{2} N2:{3}".format(quadruplets[0].shape, quadruplets[1].shape, quadruplets[2].shape, quadruplets[3].shape))
drawQuadriplets(quadruplets)
hardquadruplets = get_batch_hard(50,1,1,network4,metric_network4,dataset_train)
print("Shapes in the hardbatch 4x A:{0} P:{1} N:{2}".format(hardquadruplets[0].shape, hardquadruplets[1].shape, hardquadruplets[2].shape, hardquadruplets[3].shape))
drawQuadriplets(hardquadruplets)

def reloadFromIteration(n):
    global n_iteration
    n_iteration = n
    #network3_train.load_weights('{1}3x-temp_weights_{0:08d}.h5'.format(n_iteration,model_path))
    network4_train.load_weights('{1}4x-temp_weights_{0:08d}.h5'.format(n_iteration,model_path))



# Hyper parameters
evaluate_every = 1000 # interval for evaluating on one-shot tasks
n_iter = 10000 # No. of training iterations
log_every = 50
sample_batch_size = 16

##import wandb
#wandb.init(project=projectName)
#wandb.config.alpha1 = alpha1
#wandb.config.alpha2 = alpha2
#wandb.config.gamma=gamma
#wandb.config.ramda=ramda
#wandb.config.t1=t1
#wandb.config.t2=t2
#wandb.config.sample_batch_size = sample_batch_size
#wandb.config.learningrate = K.eval(optimizer4.lr)
#
print("Starting training process!")
print("-------------------------------------")
t_start = time.time()
for i in range(1, n_iter+1):
    microtask_start =time.time()
    quadruplets = get_batch_hardOptimized(100,16,16,network4,metric_network4, dataset_train)
    timetogetbatch = time.time()-microtask_start
    microtask_start =time.time()
    #loss1 = network3_train.train_on_batch(triplets, None)
    timebatch3 = time.time()-microtask_start
    microtask_start =time.time()
    loss2 = network4_train.train_on_batch(quadruplets, None)
    timebatch4 = time.time()-microtask_start
    n_iteration += 1
    if i % log_every == 0:
        wandb.log({'loss4x': loss2}, step=n_iteration)
    if i % evaluate_every == 0:
        elapsed_minutes = (time.time()-t_start)/60.0
        rate = i/elapsed_minutes
        eta = datetime.now()+timedelta(minutes=(n_iter-i)/rate)
        eta = eta + timedelta(hours=0) #french time
        print("[{4}] iteration {0}: {1:.1f} iter/min, Train Loss: {2}, eta : {3}".format(i, rate,loss2,n_iteration,eta.strftime("%Y-%m-%d %H:%M:%S") ))
        #network3_train.save_weights('{1}3x-temp_weights_{0:08d}.h5'.format(n_iteration,model_path))
        network4_train.save_weights('{1}4x-temp_weights_{0:08d}.h5'.format(n_iteration,model_path))
#Final save
#network3_train.save_weights('{1}3x-temp_weights_{0:08d}.h5'.format(n_iteration,model_path))
network4_train.save_weights('{1}4x-temp_weights_{0:08d}.h5'.format(n_iteration,model_path))
print("Done !")
