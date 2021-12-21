# Random Transformations

This defense randomly perturbs the inputs at training time before classification.


## Defense Idea

Instead of classifying the exact input that is provided by the adversary, this defense
queries the model on (training-time) augmented versions of the image. This hopefully
makes any small imperceptable modifications made by an adversary no longe useful
as adversarial examples.


## Training

We train our network on the same augmentations that will be used at test-time. Doing
this actually improves clean accuracy because the model is less overfit.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 71% attack succes rate


## References

### Defenses

While not exactly what this defense does, applying input-space randomization does improve robustness

Lecuyer et al. 2018 "Certified Robustness to Adversarial Examples with Differential Privacy."
http://arxiv.org/abs/1802.03471


### Attacks

Attacking randomized classifiers with an expectation over transformations can be found here

Athalye et al. 2018 "Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples."
http://arxiv.org/abs/1802.00420