# Jump Activation function

This defense changes the activation function to a new discontinuous Jump function.


## Defense Idea

Some say that adversarial examples exist because classifiers are piecewise
continuous, and therefore must interpolate between regions that don't make
sense visually. A discontinuous function would be able to better model the
space and prevent adversarial examples. The Jump activation function is like
ReLU, but has a discontinuity at zero:

jump(x) = 0 if x < 0, otherwise 5 + x

Classifiers trained with this jump function appear robust.


## Training

We train the model exactly as normal but with this new jump function. In order
to make training converge, we increase the jump constant from 0 up to 5
throughout the course of training.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 92% attack succes rate


## References

### Defenses

This defense idea is taken from an unpublished manuscript by Ian Goodfellow
developed in 2018.


### Attacks

No published attacks on this particular activation function exist.