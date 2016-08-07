#!/usr/bin/env python2

## @file
#  CNN generate file

from __future__ import division, print_function, absolute_import

from numpy import zeros, newaxis
import threading
import struct
import numpy as np
import tensorflow as tf
import specest
import time

from multiprocessing import Process, Queue

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_config
from keras import backend as K
from keras.preprocessing.image import *

from tensorflow.contrib.session_bundle import exporter

from data_generate import *

np.set_printoptions(threshold=np.nan)

## Handles flow graph for CNN
class cnn_generate(gr.top_block):

    def __init__(self, modulation, sn, sym):
    
        self.samp_rate = samp_rate = 100e3
        gr.top_block.__init__(self)

        create_blocks(self,modulation,sym,sn)

        self.blocks_add_xx_1 = blocks.add_vcc(1)
        self.blocks_multiply_const_vxx_3 = blocks.multiply_const_vcc(
            (SNRV[sn][0], ))
        self.blocks_throttle_0 = blocks.throttle(
            gr.sizeof_gr_complex * 1, samp_rate, True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(
            gr.sizeof_gr_complex * 1, 1024)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float * 1)
        self.analog_noise_source_x_0 = analog.noise_source_c(
            analog.GR_GAUSSIAN, SNRV[sn][1], 0)
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int,
                                                                   np.random.randint(0, 256, 2000000)), False)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(
            gr.sizeof_gr_complex  * 1,128)
        self.blocks_probe_signal_vx_0 = blocks.probe_signal_vc(128)
        
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
            
            self.connect((self.blocks_add_xx_1, 0), (self.blocks_stream_to_vector_0, 0))
        else:

            self.connect((self.blocks_multiply_const_vxx_3, 0),
                     (self.channels_channel_model_0 , 0))
            self.connect((self.channels_channel_model_0 , 0),
                    (self.blocks_stream_to_vector_0, 0))


        self.connect((self.blocks_stream_to_vector_0, 0),
                     (self.blocks_probe_signal_vx_0, 0))

## Invokes flow graph and returns 128 blocks of samples for the CNN
def process(train, m, sn, z, qu,sym):

    if train:
        inp = []
        out = []
    else:
        inp = [[] for k in range(0, len(SNR))]
        out = [[] for k in range(0, len(SNR))]

    tb = cnn_generate(m, sn, sym)
    tb.start()

    time.sleep(1)
    count = 0

    while True:
        o = [[],[]]

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

        if count > 500:
            tb.stop()
            break

        count += 1

    qu.put((inp, out))

## Generate CNN from training data
def cnn(train_i, train_o, test_i, test_o):
    print("About to train")

    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(1)
   
    print("Created session")
    
    batch_size = 1024
    nb_classes = len(MOD)
    nb_epoch = 2

    X_train = train_i
    Y_train = train_o

    print("About to create model")

    model = Sequential()

    model.add(Convolution2D(256, 1, 3,
                            subsample=(1, 1),
                            border_mode='valid',
                            input_shape=(1, 2, 128)))
    model.add(Activation('relu'))

    model.add(Convolution2D(80, 2, 3))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(1 - 0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax', name="out"))

    print("Going to compile model")

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("Image generator")

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=[0,1.3],
        horizontal_flip=True)

    datagen.fit(X_train)

    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=1024,shuffle=True),
                    samples_per_epoch=len(X_train), nb_epoch=5,verbose=1,validation_data=(test_i[0], test_o[0]))

    #model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
    #          verbose=1,shuffle=True, validation_data=(test_i[0], test_o[0]))

    for s in range(len(test_i)):
        X_test = test_i[s]
        Y_test = test_o[s]
        score = model.evaluate(X_test, Y_test, verbose=0)
        print("SNR",SNR[s],"Test accuracy:", score[1])

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
        input_tensor=model.input, scores_tensor=model.output)
    model_exporter.init(
        sess.graph.as_graph_def(),
        default_graph_signature=signature)
    model_exporter.export(export_path, tf.constant(export_version), sess)


load = False

if __name__ == '__main__':
    test_i, test_o = getdata(range(9),[8,16],process)
    train_i, train_o = getdata(range(9),[8,16],process,True)
        
    cnn(train_i, train_o, test_i, test_o)
