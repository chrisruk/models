#!/usr/bin/python2

from __future__ import division, print_function, absolute_import
import tensorflow as tf   
import cPickle 
from tensorflow.contrib.session_bundle import manifest_pb2
from tensorflow.contrib.session_bundle import constants
from tensorflow.contrib.session_bundle import session_bundle
from tensorflow.contrib.session_bundle import bundle_shim
import pmt
import numpy as np
from gnuradio import gr
import tensorflow as tf
from numpy import zeros, newaxis
import collections

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

def load_graph(output_graph_path):

    #sess, meta_graph_def = session_bundle.load_session_bundle_from_path(output_graph_path)
    sess, meta_graph_def = bundle_shim.load_session_bundle_or_saved_model_bundle_from_path(output_graph_path)

    with sess.as_default():
        collection_def = meta_graph_def.collection_def
        signatures_any = collection_def[
        constants.SIGNATURES_KEY].any_list.value
        signatures = manifest_pb2.Signatures()
        signatures_any[0].Unpack(signatures)
        default_signature = signatures.default_signature
        input_name = default_signature.classification_signature.input.tensor_name
        output_name = default_signature.classification_signature.scores.tensor_name
        classes = default_signature.classification_signature.classes.tensor_name
        classes = sess.run(sess.graph.get_tensor_by_name(classes))
        return (sess, input_name, output_name,classes)

sess,inp,out,classes = load_graph("/tmp/cnn/00000001/")

for t in tf.get_default_graph().as_graph_def().node:
    print(t.name)

X_train,Y_train,X_test,Y_test,mods,snrs,train_idx,test_idx,lbl = loadRadio()

acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat  = []
    
    for v in test_X_i:
        o = sess.run(out,feed_dict={inp: v})
        test_Y_i_hat.append(o)
    
    test_Y_i_hat = np.array(test_Y_i_hat)
    
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)


