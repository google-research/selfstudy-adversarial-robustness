# Majority Vote

This defense computes a best-2-of-3 majorty vote.


## Defense Idea

Because adversarial examples are often defense-specific, one way to increase
robustness is to take a majority vote among several different classifiers.
This defense trains three different models, and returns the output as the
majority prediction. If there is no majoiryt, the example is rejected.


## Training

We train the model exactly as normal but repeat the training three times.
To marginally increase robustness we additionally augment the training data
with Gaussian noise.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 62% attack succes rate


## References

### Defenses

Surprisingly we are not aware of any defenses that have this exact formulation
(if there is one please submit a PR and we will add it).

### Attacks

The field of transferability is very related to these attacks, for example
the following papers may help

Tramer et al. 2017. "The Space of Transferable Adversarial Examples"
http://arxiv.org/abs/1704.03453

Liu et al. 2016. "Delving into Transferable Adversarial Examples and Black-box Attacks."
http://arxiv.org/abs/1611.02770
