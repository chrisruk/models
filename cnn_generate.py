#!/usr/bin/env python2

## @file
#  CNN generate file
# Model is now based on https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import time
import random
import cPickle
import struct
import os
import keras

import matplotlib.pyplot as plt

from keras.optimizers  import Adam
from keras.constraints import MaxNorm

import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam


from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from tensorflow.contrib.session_bundle import exporter

from data_generate import *

np.set_printoptions(threshold=np.nan)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

## \brief Loads RadioML data
def loadRadio():
    # Load the dataset ...
    #  You will need to seperately download or generate this file
    Xd = cPickle.load(open("RML2016.10a_dict.dat",'rb'))
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
    X = np.vstack(X)

    # Partition the data
    #  into training and test sets of the form we can train/test on 
    #  while keeping SNR and Mod labels handy for each
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.5)
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    X_train = X[train_idx]
    X_test =  X[test_idx]
    Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

    return X_train,Y_train,X_test,Y_test,mods,snrs,train_idx,test_idx,lbl


## Handles flow graph for CNN
class cnn_generate(gr.top_block):

    ## \brief Creates flow graph
    ## \param modulation Modulation scheme to use
    ## \param sn List of SNRs
    ## \param sym List of symbol rates
    ## \param train Whether we are generating training data or testing data
    def __init__(self, modulation, sn, sym, train):
        self.samp_rate = samp_rate = 100e3
        gr.top_block.__init__(self)

        create_blocks(self, modulation, sym, sn, train)

        self.blocks_add_xx_1 = blocks.add_vcc(1)
        self.blocks_multiply_const_vxx_3 = blocks.multiply_const_vcc(
            (SNRV[sn][0], ))
        self.blocks_throttle_0 = blocks.throttle(
            gr.sizeof_gr_complex * 1, samp_rate, True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(
            gr.sizeof_gr_complex * 1, 1024)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float * 1)
        self.analog_noise_source_x_0 = analog.noise_source_c(
            analog.GR_GAUSSIAN, SNRV[sn][1], -struct.unpack(">L", os.urandom(4))[0])
        self.analog_random_source_x_0 = blocks.vector_source_b(
            map(int, np.random.randint(0, 256, 2000000)), False)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(
            gr.sizeof_gr_complex * 1, 128)
        self.blocks_probe_signal_vx_0 = blocks.probe_signal_vc(128)


        self.blocks_keep_m_in_n_0_0 = blocks.keep_m_in_n(gr.sizeof_gr_complex, 128, (128*2), 64)
        self.blocks_keep_m_in_n_0 = blocks.keep_m_in_n(gr.sizeof_gr_complex, 128, (128*2), 0)

        self.blocks_keepv1 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 128)
        self.blocks_keepv2 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 128)

        self.blocks_interleave_0 = blocks.interleave(gr.sizeof_gr_complex*128, 1)

        if not channel_model:

            self.connect((self.analog_noise_source_x_0, 0),
                         (self.blocks_add_xx_1, 1))
            self.connect((self.blocks_multiply_const_vxx_3, 0),
                         (self.blocks_add_xx_1, 0))

        if modulation == "wbfm":
            self.connect((self.blocks_wavfile_source_0, 0),
                         (self.analog_wfm_tx_0, 0))
            self.connect((self.analog_wfm_tx_0, 0),
                         (self.blocks_throttle_0, 0))
        elif modulation == "nfm":
            self.connect((self.blocks_wavfile_source_0, 0),
                         (self.analog_nfm_tx_0, 0))
            self.connect((self.analog_nfm_tx_0, 0),
                         (self.blocks_throttle_0, 0))
        else:
            self.connect((self.analog_random_source_x_0, 0),
                         (self.digital_mod, 0))
            self.connect((self.digital_mod, 0), (self.blocks_throttle_0, 0))


        self.connect((self.blocks_throttle_0, 0),
                     (self.rational_resampler_xxx_0, 0))

        self.connect((self.rational_resampler_xxx_0, 0),
                     (self.blocks_multiply_const_vxx_3, 0))

        if not channel_model:
            self.connect((self.blocks_add_xx_1, 0), (self.blocks_keep_m_in_n_0, 0))    
            self.connect((self.blocks_add_xx_1, 0), (self.blocks_keep_m_in_n_0_0, 0))   

            self.connect((self.blocks_keep_m_in_n_0, 0), (self.blocks_keepv1, 0))    
            self.connect((self.blocks_keep_m_in_n_0_0, 0), (self.blocks_keepv2, 0))  

            self.connect((self.blocks_keepv1, 0), (self.blocks_interleave_0, 0))    
            self.connect((self.blocks_keepv2, 0), (self.blocks_interleave_0, 1))    

            self.connect((self.blocks_interleave_0, 0),
                     (self.blocks_probe_signal_vx_0, 0))
        else:
            self.connect((self.blocks_multiply_const_vxx_3, 0),
                         (self.channels_channel_model_0, 0))

            self.connect((self.channels_channel_model_0, 0), (self.blocks_keep_m_in_n_0, 0))    
            self.connect((self.channels_channel_model_0, 0), (self.blocks_keep_m_in_n_0_0, 0))   

            self.connect((self.blocks_keep_m_in_n_0, 0), (self.blocks_keepv1, 0))    
            self.connect((self.blocks_keep_m_in_n_0_0, 0), (self.blocks_keepv2, 0))  

            self.connect((self.blocks_keepv1, 0), (self.blocks_interleave_0, 0))    
            self.connect((self.blocks_keepv2, 0), (self.blocks_interleave_0, 1))    

            self.connect((self.blocks_interleave_0, 0),
                     (self.blocks_probe_signal_vx_0, 0))

## \brief Invokes flow graph and returns 128 blocks of samples for the CNN
## \param train Whether we are training or not
## \param m Modulation scheme
## \param sn List of SNRs
## \param z One-hot array representing modulation scheme
## \param qu Queue to return data 
## \param sym List of symbol rates
def process(train, m, sn, z, qu, sym):

    # Without this, multiple processes all generate exactly the same sequence of random numbers
    reseed()

    if train:
        inp = []
        out = []
    else:
        inp = [[] for k in range(0, len(SNR))]
        out = [[] for k in range(0, len(SNR))]

    tb = cnn_generate(m, sn, sym, train)
    tb.start()

    time.sleep(1)
    count = 0

    while True:
        o = [[], []]

        floats = tb.blocks_probe_signal_vx_0.level()
        for v in floats:
            o[0].append(v.real)
            o[1].append(v.imag)

        if train:
            inp.append(np.array([o]))
            out.append(np.array(z))
        else:
            inp[sn].append(np.array([o]))
            out[sn].append(np.array(z))

        if count > 100:
            tb.stop()
            break

        count += 1

    qu.put((inp, out))

## \brief Generate CNN from training data
## \param train_i Training data
## \param train_o Class for each training item
## \param test_i Testing data
## \param test_o Class for each testing item
## \param mod List of modulation schemes
def cnn(train_i, train_o, test_i, test_o,mods,snrs,train_idx,test_idx,lbl):
 
    # CNN1
    c1 = 64
    c2 = 16
    dl = 128
    
    """ 
    # CNN2
    c1 = 256
    c2 = 80
    dl = 256
    """
    
    nb_epoch = 400

    sess = tf.Session()

    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    K.set_session(sess)
    K.set_learning_phase(1)
    
    classes = mods
    #X_train,Y_train = shuffle_in_unison_inplace( np.array(train_i) , np.array(train_o) )

    X_train = train_i
    Y_train = train_o
    X_test = test_i
    Y_test = test_o

    in_shp = list(X_train.shape[1:])

    
    # Build VT-CNN2 Neural Net model using Keras primitives -- 
    #  - Reshape [N,2,128] to [N,1,2,128] on input
    #  - Pass through 2 2DConv/ReLu layers
    #  - Pass through 2 Dense layers (ReLu and Softmax)
    #  - Perform categorical cross entropy optimization

    dr = 0.5 # dropout rate (%)
    model = models.Sequential()
    model.add(Reshape([1]+in_shp, input_shape=in_shp))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(256, 1, 3, border_mode='valid', activation="relu", init='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Convolution2D(80, 2, 3, border_mode="valid", activation="relu",  init='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', init='he_normal'))
    model.add(Dropout(dr))
    model.add(Dense( len(classes), init='he_normal' ))
    model.add(Activation('softmax',name="out"))
    model.add(Reshape([len(classes)]))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()

    #datagen = ImageDataGenerator()
        #featurewise_center=False,
        #featurewise_std_normalization=False,
        #rotation_range=0,
        #width_shift_range=0.3,
        #height_shift_range=0.3,
        #zoom_range=[0,1.3],
        #shear_range=0.2,
        # horizontal_flip=True,
        #vertical_flip=True)

    """
    datagen.fit(X_train)

    model.fit_generator(
        datagen.flow(
            X_train,
            Y_train,
            batch_size=1024,
            shuffle=True),
        samples_per_epoch=len(X_train),
        nb_epoch=nb_epoch,
        verbose=1,
        validation_data=(
            test_i[0],
            test_o[0]))
    """
    # Set up some params 
    nb_epoch = 25 #100   # number of epochs to train on
    batch_size = 1024  # training batch size

    tb = TensorBoard(log_dir='./logs')

    # perform training ...
    #   - call the main training loop in keras for our network+dataset
    filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
    history = model.fit(X_train,
        Y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        show_accuracy=False,
        verbose=2,
        validation_data=(X_test, Y_test),
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        ])
    # we re-load the best weights once training is finished
    model.load_weights(filepath)

    K.set_learning_phase(0)



    acc = {}
    for snr in snrs:

        # extract classes @ SNR
        test_SNRs = map(lambda x: lbl[x][1], test_idx)
        test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        #print("PREDICT ",test_Y_i_hat)

        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])
        for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            conf[j,k] = conf[j,k] + 1
        for i in range(0,len(classes)):
            confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        plt.figure()
        plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print ("Overall Accuracy: ", cor / (cor+ncor))
        acc[snr] = 1.0*cor/(cor+ncor)


    config = model.get_config()
    weights = model.get_weights()

    new_model = models.Sequential.from_config(config)
    new_model.set_weights(weights)

    export_path = "/tmp/cnn"
    export_version = 1

    labels_tensor = tf.constant(mods)

    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    signature = exporter.classification_signature(
        input_tensor=new_model.input,classes_tensor=labels_tensor, scores_tensor=new_model.output)
    model_exporter.init(
        sess.graph.as_graph_def(),
        default_graph_signature=signature)
    model_exporter.export(export_path, tf.constant(export_version), sess)

if __name__ == '__main__':
    
    """
    reseed()
    test_i, test_o = getdata(range(1), [3,4], process)
    reseed()
    train_i, train_o = getdata(range(9), [3,4], process, True)
    """

    X_train,Y_train,X_test,Y_test,mods,snrs,train_idx,test_idx,lbl = loadRadio()
    
    cnn(X_train,Y_train,X_test,Y_test,mods,snrs,train_idx,test_idx,lbl)
