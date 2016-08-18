#!/usr/bin/env python2

## @file
#  CNN generate file

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import time
import random
import cPickle
import struct
import os

from keras.optimizers  import Adam
from keras.constraints import MaxNorm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from tensorflow.contrib.session_bundle import exporter

from data_generate import *

np.set_printoptions(threshold=np.nan)


def reseed():
    random.seed()
    np.random.seed() 

def loadRadio():

    radioml = cPickle.load(open("2016.04C.multisnr.pkl",'rb'))

    data = {}
    allm = []

    for k in radioml.keys():
        data[k[0]] = {}
        allm.append(k[0])

    mod = sorted(set(allm))

    for m in mod:
        dat = []
        for k in radioml.keys():
            if k[0] == m :
                for sig in range(len(radioml[k])):
                    a = np.array(radioml[k][sig][0])
                    b = np.array(radioml[k][sig][1])
                    if k[1] not in data[k[0]]:
                        data[k[0]][k[1]] = []
                    data[k[0]][k[1]].append([[a,b]])
    
    X = []
    Y = []
    x = {}
    y = {} 

    mval = {}
    count = 0

    for m in mod:
        z = np.zeros((len(mod),))
        z[count] = 1     
        mval[m] = z
        for snr in data[m]:

            dat = data[m][snr]
            for d in dat[:int(len(dat)/2)]:

                X.append(d)
                Y.append(z)

            for d in dat[int(len(dat)/2):]:

                if not snr in x:
                    x[snr] = []
                    y[snr] = []

                x[snr].append(d)
                y[snr].append(z)

        count += 1       

    return X,Y,x,y,mod,data


## Handles flow graph for CNN
class cnn_generate(gr.top_block):

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

            #self.connect((self.blocks_add_xx_1, 0),
            #             (self.blocks_stream_to_vector_0, 0))


            self.connect((self.blocks_add_xx_1, 0), (self.blocks_keep_m_in_n_0, 0))    
            self.connect((self.blocks_add_xx_1, 0), (self.blocks_keep_m_in_n_0_0, 0))   

            self.connect((self.blocks_keep_m_in_n_0, 0), (self.blocks_keepv1, 0))    
            self.connect((self.blocks_keep_m_in_n_0_0, 0), (self.blocks_keepv2, 0))  

            self.connect((self.blocks_keepv1, 0), (self.blocks_interleave_0, 0))    
            self.connect((self.blocks_keepv2, 0), (self.blocks_interleave_0, 1))    

            self.connect((self.blocks_interleave_0, 0),
                     (self.blocks_probe_signal_vx_0, 0))


            

        else:

            print("Using channel model")

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




        #self.connect((self.blocks_stream_to_vector_0, 0),
        #             (self.blocks_probe_signal_vx_0, 0))

## Invokes flow graph and returns 128 blocks of samples for the CNN
def process(train, m, sn, z, qu, sym):

    # Without this, multiple processes all generate exactly the same sequence of random numbers
    reseed()

    #print(np.random.randint(0, 100, 10))

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



def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

## Generate CNN from training data
def cnn(train_i, train_o, test_i, test_o):
    
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


    print("About to train")

    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(1)
    
    print("Created session")

    nb_classes =  11# len(MOD) # 11

    #X_train,Y_train = shuffle_in_unison_inplace( np.array(train_i) , np.array(train_o) )
    X_train = np.array(train_i)
    Y_train = np.array(train_o)

    print("About to create model")

    model = Sequential()

    model.add(Convolution2D(c1, 1, 3,
                            #subsample=(1, 1),
                            #border_mode='valid',
                            input_shape=(1, 2, 128)))
                            #W_regularizer = l2(.01))) #,W_constraint = MaxNorm(2)))
    model.add(Activation('relu'))

    model.add(Convolution2D(c2, 2, 3)) #W_regularizer = l2(.01))) #,W_constraint = MaxNorm(2)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(dl))
    model.add(Activation('relu'))
    model.add(Dropout(1 - 0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax', name="out"))

    print("Going to compile model")

    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print("Image generator, train",len(X_train))
  
     
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

    #print("len",X_train.shape,Y_train.shape)#,test_i[18].shape,test_o[18].shape)

    model.fit(X_train, Y_train, batch_size=1024, nb_epoch=nb_epoch,
            verbose=1,shuffle=True, validation_split=0.1)#validation_data=(np.array(test_i[18]), np.array(test_o[18])))
    
    #learning = sess.graph.get_tensor_by_name("keras_learning_phase:0")

    for s in sorted(test_i):
        X_test = np.array(test_i[s])
        Y_test = np.array(test_o[s])
        score = model.evaluate(X_test, Y_test, verbose=0)
        print("SNR", s, "Test accuracy:", score[1])

    K.set_learning_phase(0)

    config = model.get_config()
    weights = model.get_weights()

    new_model = Sequential.from_config(config)
    new_model.set_weights(weights)

    export_path = "/tmp/cnn"
    export_version = 1

    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    signature = exporter.classification_signature(
        input_tensor=new_model.input, scores_tensor=new_model.output)
    model_exporter.init(
        sess.graph.as_graph_def(),
        default_graph_signature=signature)
    model_exporter.export(export_path, tf.constant(export_version), sess)

if __name__ == '__main__':

    
    reseed()

    #test_i, test_o = getdata(range(1), [3,4], process)
        
    time.sleep(10)

    # This is very important!
    reseed()

    #train_i, train_o = getdata(range(9), [3,4], process, True)
    
    train_i,train_o,test_i,test_o,mod,data = loadRadio()
    #print("train",train_i[0],train_o[0])
    #print("test",test_i[18][0],test_o[18][0])
 
    cnn(train_i, train_o, test_i, test_o)
