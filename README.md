# models

Raw CNN &amp; FAM model generation

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

I'm currently investigating an issue I've found with the Keras implementation, of both these models, where 
there is seemingly an 8% reduction in accuracy to my TFlearn implementation with RadioML data for the raw CNN model, however
I believe this reduction could apply to both models.
