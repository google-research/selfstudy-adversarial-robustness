# Ensemble of Binary Classifiers

This defense implements an ensemble of binary classifiers.


## Defense Idea

Instead of training a single classifier to predict if an input is a given
class, we train 10 binary classifiers, each of which makes the binary decision
"is this input class X?"

The final prediction is made by merging each of these classifiers in a way
that follows the suggetions from prior work: we first convert the output
with a sigmoid to a yes/no decision for each clas, and then take the softmax
output over these sigmoid values.


## Training

Training is identical to standard training, but with N different classifiers
trained on (imbalanced) datasets containing 5,000 correct instances and 45,000
incorrect instances. The output is trained with a softmax output loss.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 90% attack succes rate


## References



### Defenses

This defense has been suggested many times in the past, however the most
direct influence of this idea is the paper


Verma & Swami 2019. "Error Correcting Output Codes Improve Probability Estimation and Adversarial Robustness of Deep Neural Networks"
https://proceedings.neurips.cc/paper/2019/hash/cd61a580392a70389e27b0bc2b439f49-Abstract.html


### Attacks

An attack on the above proposed paper is available in

Tramer et al. 2020 "On Adaptive Attacks to Adversarial Example Defenses"
http://arxiv.org/abs/2002.08347