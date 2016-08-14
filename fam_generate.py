#!/usr/bin/env python2

## @file
#  FAM generate file

from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import specest
import time
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import *
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from tensorflow.contrib.session_bundle import exporter

from data_generate import *

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def reseed():
    #random.seed()
    np.random.seed() 

Np = 64  # 2xNp is the number of columns
P =  256  # number of new items needed to calculate estimate
L =  2

np.set_printoptions(threshold=np.nan)

## Handles flowgraph for FAM
class fam_generate(gr.top_block):

    def __init__(self, modulation, sn, sym, train):

        self.samp_rate = samp_rate = 100e3
        gr.top_block.__init__(self)

        create_blocks(self, modulation, sym, sn, train)

        self.blocks_add_xx_1 = blocks.add_vcc(1)
        self.specest_cyclo_fam_1 = specest.cyclo_fam(Np, P, L)
        self.blocks_multiply_const_vxx_3 = blocks.multiply_const_vcc(
            (SNRV[sn][0], ))
        self.blocks_throttle_0 = blocks.throttle(
            gr.sizeof_gr_complex * 1, samp_rate, True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(
            gr.sizeof_gr_complex * 1, 1024)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float * 1)
        self.analog_noise_source_x_0 = analog.noise_source_c(
            analog.GR_GAUSSIAN, SNRV[sn][1], np.random.randint(np.iinfo(np.int32).max))
        self.analog_random_source_x_0 = blocks.vector_source_b(
            map(int, np.random.randint(0, 256, 2000000)), False)
        self.msgq_out = blocks_message_sink_0_msgq_out = gr.msg_queue(1)
        self.blocks_message_sink_0 = blocks.message_sink(
            gr.sizeof_float * 2 * Np, blocks_message_sink_0_msgq_out, False)
        self.blocks_vector_to_stream_0 = blocks.vector_to_stream(
            gr.sizeof_float * 1, 2 * Np)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(
            gr.sizeof_float * 1, 2 * P * L * ((2 * Np) - 0))
        self.blocks_probe_signal_vx_0 = blocks.probe_signal_vf(
            2 * P * L * ((2 * Np) - 0))
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
        self.connect((self.blocks_add_xx_1, 0), (self.specest_cyclo_fam_1, 0))
        self.connect((self.specest_cyclo_fam_1, 0),
                     (self.blocks_vector_to_stream_0, 0))
        self.connect((self.blocks_vector_to_stream_0, 0),
                     (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0),
                     (self.blocks_probe_signal_vx_0, 0))

## Invokes flow graph and returns FAM data
def process(train, m, sn, z, qu, sym):

    # Without this, multiple processes all generate exactly the same sequence of random numbers
    reseed()

    if train:
        inp = []
        out = []
    else:
        inp = [[] for k in range(0, len(SNR))]
        out = [[] for k in range(0, len(SNR))]

    tb = fam_generate(m, sn, sym, train)
    tb.start()

    time.sleep(3)
    count = 0

    while True:
        floats = tb.blocks_probe_signal_vx_0.level()

        if np.sum(floats) == 0:
            print("Found empty FAM")
            continue

        floats = (floats - np.mean(floats)) / np.std(floats)
        floats = np.reshape(floats, (2 * P * L, (2 * Np) - 0))

        if train:
            inp.append(np.array([floats]))
            out.append(np.array(z))
        else:
            inp[sn].append(np.array([floats]))
            out[sn].append(np.array(z))

        if count > 30:
            tb.stop()
            break

        count += 1

    qu.put((inp, out))

## Generate CNN from training data
def fam(train_i, train_o, test_i, test_o):
    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(1)

    batch_size = 60
    nb_classes = len(MOD)
    nb_epoch = 10

    img_rows, img_cols = 2 * P * L, 2 * Np
    nb_filters = 96
    nb_pool = 2

    X_train,Y_train = shuffle_in_unison_inplace( np.array(train_i) , np.array(train_o) )

    model = Sequential()
    model.add(Convolution2D(64, 11, 11,subsample=(2,2),
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5)) 

    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5)) 

    model.add(Dense(nb_classes,init='normal'))
    model.add(Activation('softmax', name="out"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    """
    datagen = ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        rotation_range=20,
        #width_shift_range=0.3,
        #height_shift_range=0.3,
        #zoom_range=[0,1.3],
        horizontal_flip=True,
        vertical_flip=True)

    datagen.fit(X_train)

    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size,shuffle=True),
                    samples_per_epoch=len(X_train), nb_epoch=5,verbose=1,validation_data=(test_i[0], test_o[0]))

    """

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, shuffle=True, validation_data=(test_i[0], test_o[0]))


    for s in range(len(test_i)):
        if len(test_i[s]) == 0:
            continue
        X_test = test_i[s]
        Y_test = test_o[s]
        score = model.evaluate(X_test, Y_test, verbose=0)
        print("SNR", SNR[s], "Test accuracy:", score[1])

    K.set_learning_phase(0)
    config = model.get_config()
    weights = model.get_weights()

    new_model = Sequential.from_config(config)
    new_model.set_weights(weights)

    export_path = "/tmp/fam"
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

    test_i, test_o = getdata(range(1), [3], process)

    reseed()

    train_i, train_o = getdata(range(5), [3], process, True)

    fam(train_i, train_o, test_i, test_o)
