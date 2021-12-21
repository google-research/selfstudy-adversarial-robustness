# Discretization Defense

This defense randomly perturbs neurons during classification.


## Defense Idea

During the forward-pass of the model, instead of using the exact network activations
as computed normally, we drop a subset of these weights at random. Larger weights
are removed with higher probability, and the remaining weights are scaled up to
compensate as done in dropout.


## Training

We use an undefended network without modification for the attack.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 80% attack succes rate


## References

### Defenses

The two most closely related defenses are


Dhillon et al. 2018 "Stochastic Activation Pruning for Robust Adversarial Defense."
http://arxiv.org/abs/1803.01442

Xiao et al. 2019 "Enhancing Adversarial Defense by k-Winners-Take-All."
http://arxiv.org/abs/1905.10510


### Attacks

The most direct attack results can be found in

Athalye et al. 2018 "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples."
http://arxiv.org/abs/1802.00420

http://arxiv.org/abs/2010.00071
Erratum Concerning the Obfuscated Gradients Attack on Stochastic Activation Pruning.
Guneet S. Dhillon; Nicholas Carlini