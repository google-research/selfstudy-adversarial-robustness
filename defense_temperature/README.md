# Baseline Model

This defense trains a neural network with significantly higher "softmax
temperature" in order to flatten the loss function.


## Defense Idea

Given an input x to a classifier, before classifying it we drop the low-3 bits
and then feed the (quantized) input into the neural network.


## Training

The model is trained using almost exactly the code used to train the baseline
neural network. Given the training dataset, we first perform the bit depth
quantization and train the model on this quantized data. This helps to increase
the accuracy of the model on quantized test examples.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: TODO attack success rate
- l_2 distortion of TODO: TODO attack success rate


## References

### Defenses

This bit-depth quantization idea has been published many times in the past.
The earliest paper which proposed (something like) it is TODO.

### Attacks

TODO