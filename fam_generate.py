#!/usr/bin/env python2
from __future__ import division, print_function, absolute_import

from gnuradio import filter
from gnuradio import gr
from gnuradio import audio, analog
from gnuradio import digital
from gnuradio import blocks
from grc_gnuradio import blks2 as grc_blks2

from numpy import zeros, newaxis
import threading
import numpy
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

from tensorflow.contrib.session_bundle import exporter


snr = ["20","15","10","5","0","-5","-10","-15","-20"] 
snrv = [[1,0.32],[1,0.435],[1,0.56],[1,0.75],[1,1],[0.75,1],[0.56,1],[0.435,1],[0.32,1]]
mod = ["fsk","qam16","qam64","2psk","4psk","8psk","gmsk","wbfm","nfm"]

Np = 64 # 2xNp is the number of columns
P = 256 # number of new items needed to calculate estimate
L = 2

np.set_printoptions(threshold=np.nan)

class fam_generate(gr.top_block):
    def __init__(self,modulation,sn,sym):

        self.samp_rate = samp_rate = 100e3
        gr.top_block.__init__(self)

        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=2,
                decimation=1,
                taps=None,
                fractional_bw=None,
        )


        if modulation == "2psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=2,
                mod_code="gray",
                differential=True,
                samples_per_symbol=sym,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
        elif modulation == "4psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=4,
                mod_code="gray",
                differential=True,
                samples_per_symbol=sym,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
        elif modulation == "8psk":
            self.digital_mod = digital.psk.psk_mod(
                constellation_points=8,
                mod_code="gray",
                differential=True,
                samples_per_symbol=sym,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )

        elif modulation == "fsk":
            self.digital_mod = digital.gfsk_mod(
        	    samples_per_symbol=sym,
        	    sensitivity=1.0,
        	    bt=0.35,
        	    verbose=False,
        	    log=False,
            )
        elif modulation == "qam16":
            self.digital_mod = digital.qam.qam_mod(
                constellation_points=16,
                mod_code="gray",
                differential=True,
                samples_per_symbol=sym,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
        elif modulation == "qam64":
            self.digital_mod = digital.qam.qam_mod(
                constellation_points=64,
                mod_code="gray",
                differential=True,
                samples_per_symbol=sym,
                excess_bw=0.35,
                verbose=False,
                log=False,
            )
        elif modulation == "gmsk":
            self.digital_mod = digital.gmsk_mod(
        	    samples_per_symbol=sym,
        	    bt=0.35,
        	    verbose=False,
        	    log=False,
            )


        self.blocks_wavfile_source_0 = blocks.wavfile_source("/home/chris/Desktop/music.wav", False)

        self.analog_wfm_tx_0 = analog.wfm_tx(
        	audio_rate=44100,
        	quad_rate=44100*5,
        	tau=75e-6,
        	max_dev=75e3,
        	fh=-1.0,
        )

        self.analog_nfm_tx_0 = analog.nbfm_tx(
        	audio_rate=44100,
        	quad_rate=44100*2,
        	tau=75e-6,
        	max_dev=5e3,
        	fh=-1.0,
        )

        self.blocks_add_xx_1 = blocks.add_vcc(1)
        self.specest_cyclo_fam_1 = specest.cyclo_fam(Np, P, L)
        self.blocks_multiply_const_vxx_3 = blocks.multiply_const_vcc((snrv[sn][0], ))
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 1024)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float*1)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, snrv[sn][1], 0)
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, 256, 2000000)), False)
        self.msgq_out = blocks_message_sink_0_msgq_out = gr.msg_queue(1) 
        self.blocks_message_sink_0 = blocks.message_sink(gr.sizeof_float*2*Np, blocks_message_sink_0_msgq_out, False)  


        self.blocks_vector_to_stream_0 = blocks.vector_to_stream(gr.sizeof_float*1, 2*Np)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_float*1, 2*P*L*((2*Np)-0))
        self.blocks_probe_signal_vx_0 = blocks.probe_signal_vf(2*P*L*((2*Np)-0))

        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_1, 1))    
        self.connect((self.blocks_multiply_const_vxx_3, 0), (self.blocks_add_xx_1, 0)) 

        if modulation == "wbfm":
            self.connect((self.blocks_wavfile_source_0, 0), (self.analog_wfm_tx_0, 0))    
            self.connect((self.analog_wfm_tx_0, 0), (self.blocks_throttle_0, 0))  
        elif modulation == "nfm":
            self.connect((self.blocks_wavfile_source_0, 0), (self.analog_nfm_tx_0, 0))    
            self.connect((self.analog_nfm_tx_0, 0), (self.blocks_throttle_0, 0))  
        else:
            self.connect((self.analog_random_source_x_0, 0), (self.digital_mod, 0)) 
            self.connect((self.digital_mod, 0), (self.blocks_throttle_0, 0))    

        self.connect((self.blocks_throttle_0, 0),(self.rational_resampler_xxx_0, 0))    

        self.connect((self.rational_resampler_xxx_0, 0),(self.blocks_multiply_const_vxx_3, 0))    
        self.connect((self.blocks_add_xx_1, 0),(self.specest_cyclo_fam_1, 0))    

        self.connect((self.specest_cyclo_fam_1, 0), (self.blocks_vector_to_stream_0, 0))    

        self.connect((self.blocks_vector_to_stream_0, 0), (self.blocks_stream_to_vector_0, 0))    

        self.connect((self.blocks_stream_to_vector_0, 0), (self.blocks_probe_signal_vx_0, 0))    


def process(train,m,sn,z,qu):

    if train:
        inp = []
        out = []
    else:
        inp = [[] for k in range(0,len(snr))]
        out = [[] for k in range(0,len(snr))]

    print(inp,out,sn,len(inp))

    tb = fam_generate(m,sn,2)
    tb.start()

    time.sleep(1)
    count = 0
    fin = False
    old = None

    while True: 
        floats = tb.blocks_probe_signal_vx_0.level()
        floats = (floats - np.mean(floats)) / np.std(floats)
        za = np.reshape(floats, (2*P*L, (2*Np)-0)) 
        floats = np.reshape(floats, (2*P*L, (2*Np)-0)) 

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

    qu.put((inp,out))


def getdata(sn,train=False):
    mcount = 0

    if train:
        inp = []
        out = []
    else:
        inp = [[] for k in range(0,sn)]
        out = [[] for k in range(0,sn)]

    flow = [None for k in range(len(mod))]

    for m in mod:

        z = np.zeros((len(mod),))
        z[mcount] = 1  
        
        print("MOD ",z)  
    
        q = Queue() #create a queue object
        plist = [] 
        for s in range(0,sn):
            p = Process(target=process, args=(train,m,s,z,q))
            plist.append(p)
            p.start()

        for p in plist:
            job = q.get()
            if train:
                inp += job[0]
                out += job[1]
            else:
                print(len(inp),len(job[0]),sn)
                for i in range(len(inp)):
                    inp[i] += job[0][i]
                    out[i] += job[1][i]
            
        for p in plist:
            p.join()

            
        mcount += 1

    return np.array(inp),np.array(out)


def cnn(train_i,train_o,test_i,test_o):
    sess = tf.Session()

    K.set_session(sess)

    K.set_learning_phase(1)  # all new operations will be in test mode from now on

    batch_size = 60
    nb_classes = len(mod)
    nb_epoch = 30

    img_rows, img_cols = 2*P*L, 2*Np
    nb_filters = 64
    nb_pool = 2

    X_train = train_i
    Y_train = train_o

    X_test = test_i[0]
    Y_test = test_o[0]

    model = Sequential()

    model.add(Convolution2D(nb_filters, 3, 3,
                        subsample=(2,2),
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Convolution2D(nb_filters, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(1-0.7))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(1-0.7))
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax',name="out"))

    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)# validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    K.set_learning_phase(0)  # all new operations will be in test mode from now on

    # serialize the model and get its weights, for quick re-building
    config = model.get_config()
    weights = model.get_weights()

    new_model = Sequential.from_config(config)
    new_model.set_weights(weights)

    export_path = "/tmp/sess" 
    export_version = 1 

    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    signature = exporter.classification_signature(input_tensor=model.input,
                                              scores_tensor=model.output)
    model_exporter.init(sess.graph.as_graph_def(),
                    default_graph_signature=signature)
    model_exporter.export(export_path, tf.constant(export_version), sess)


load = False

if __name__ == '__main__':    
    test_i , test_o = getdata(1)
    train_i , train_o = getdata(1,True)

    cnn(train_i,train_o,test_i,test_o)

















