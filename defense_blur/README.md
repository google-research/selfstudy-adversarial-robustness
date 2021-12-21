# Blur Defense

This defense implements a blur function.


## Defense Idea

Adversarial examples are often seen as having high frequency noise.
This defense blurs the inputs to remove this noise.


## Training

Training the blur defense is straightforward: at training time, all of the
training images are blurred, so that the classifier learns to be accurate on
blurry images.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 95% attack succes rate


## References



### Defenses

Blurring is rarely proposed as a defense by itself, but this idea is often
a component of many defenses:

Das et al. 2017 "Keeping the Bad Guys Out: Protecting and Vaccinating Deep Learning with JPEG Compression."
http://arxiv.org/abs/1705.02900

Subramanya et al. 2017 "Confidence estimation in Deep Neural networks via density modelling"
http://arxiv.org/abs/1707.07013


### Attacks

While canonical attacks exist for attacking blurred classifiers in particular,
any paper discussing defense-specific adaptive attacks would be related.