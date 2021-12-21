# Discretization Defense

This defense discretizes the input before classification.


## Defense Idea

Instead of classifying inputs as continuous variables in the domain [0,1],
this defense splits each input channel into 20 buckets, where each bucket
i is set if the input is greater than i/20. So for example we represent
0.0 as [1 0 0 0 0 ... 0]
0.1 as [1 1 1 0 0 ... 0]
1.0 as [1 1 1 1 1 ... 1]


## Training

We train the model exactly as normal but with this discretization step applied
to the inputs.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 97 attack succes rate


## References

### Defenses

There are several defenses that take this form of discretization, most related


Buckman et al. 2018. "Thermometer Encoding: One Hot Way To Resist Adversarial Examples"
https://openreview.net/pdf?id=S18Su--CW

Xu et al. 2017. "Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks."
http://arxiv.org/abs/1704.01155

### Attacks

The most direct attack results can be found in

Athalye et al. 2018 "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples."
http://arxiv.org/abs/1802.00420