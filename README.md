# models

Raw CNN &amp; FAM model generation

## 6th Jan 2017

Working on updating the CNN python script, to use  https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb

I notice there is now a bug when I try to load the saved model, via GRC, which I am trying to solve.

```
handler caught exception: Attempting to use uninitialized value convolution2d_1_W_1
	 [[Node: convolution2d_1_W_1/read = Identity[T=DT_FLOAT, _class=["loc:@convolution2d_1_W_1"], _device="/job:localhost/replica:0/task:0/cpu:0"](convolution2d_1_W_1)]]

Caused by op u'convolution2d_1_W_1/read', defined at:
```

We need TensorFlow version 0.10+ to run.

Also you need Keras, which can be installed easily from pip.

It is important that .keras/keras.json, looks something like this:

```
{
    "image_dim_ordering": "th", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "tensorflow"
}
```

For fam_generate.py be sure to install the gr-specest fork from: 

https://github.com/chrisruk/gr-specest

# To generate documentation use:

doxygen models.dox

# Note

Please use the FAM model at the moment, I'm just tweaking the CNN model, as I need to implement 64 sample shifts.

## 12th Sept 2016

I'm currently investigating an issue I've found with the Keras implementations, where 
there is seemingly an 8% reduction in accuracy to my TFlearn implementation with RadioML data for the raw CNN model, however
I believe this reduction could apply to both models.
