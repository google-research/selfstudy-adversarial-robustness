# Injecting backdoor signatures

Inject backdoors into a model so that adversarial examples can be detected by
looking for the fingerprints of a backdoor. This is a very challenging defense:
we do not currently have a complete break.

## Defense Idea

Construct 10 independent backdoor patterns (bd_i) and train the model so that for
any input x we have f(x + bd_i) = i. On the training data record the expected
activation vector E_{x in X}[g(x + bd_i)] for each of the backdoor patterns.
At test time, to detect if an input is adversarial, first reject inputs where the
first property is violatd (and adding bd_i does not end up with the classification
reaching class i), and then reject inputs that are too similar to the backdoor
activation vector.


## Training

Training is simple. After constructing 10 backdoor patterns, train the neural
network so that f(x)=y is correctly predicted, but also so that f(x+bd_i)=i.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 28% attack success rate


## References

### Defenses

There are two defenses that directly inspired this design.


Dathathri et al. 2018 "Detecting Adversarial Examples via Neural Fingerprinting"
https://arxiv.org/abs/1803.03870

Shan et al. 2020 "Gotta Catch 'Em All: Using Honeypots to Catch Adversarial Attacks on Neural Networks"
https://arxiv.org/abs/1904.08554

### Attacks

Carlini "A Partial Break of the Honeypots Defense to Catch Adversarial Attacks"
https://arxiv.org/abs/2009.10975
