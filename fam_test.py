#!/usr/bin/python2
## @file
#  FAM loading file
from __future__ import division, print_function, absolute_import
import tensorflow as tf    
from tensorflow.contrib.session_bundle import manifest_pb2
from tensorflow.contrib.session_bundle import constants
from tensorflow.contrib.session_bundle import session_bundle
from fam_generate import *

## Load TensorFlow serving session from file
def load_from_file():
    sess, meta_graph_def = session_bundle.LoadSessionBundleFromPath("/tmp/fam/00000001") 

    with sess.as_default():

        test_i, test_o = getdata(range(1), [8, 16], process) 
        collection_def = meta_graph_def.collection_def
        signatures_any = collection_def[constants.SIGNATURES_KEY].any_list.value
        signatures = manifest_pb2.Signatures()
        signatures_any[0].Unpack(signatures)
        default_signature = signatures.default_signature

        input_name = default_signature.classification_signature.input.tensor_name
        output_name = default_signature.classification_signature.scores.tensor_name
    
        for s in range(len(test_i)):

            gd = 0
            z = 0
            allv = test_i[s]

            if len(allv) > 0:

                for v in allv:
                    if np.argmax(sess.run ([output_name],{input_name: [ v ]})[0]) == np.argmax(test_o[s][z]):
                        gd += 1
                    z = z + 1

                print ("SNR",SNR[s],"ACC",gd/z)

load_from_file()
