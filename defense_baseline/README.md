# Baseline Model

This defense implements a baseline (undefended) neural network. 


## Defense Idea

There is no special defense applied. The network is trained with standard
cross entropy loss directly on the output of the neural network.


## Training

The training code is written to satisfy a few objectives

- It should be simple. Any code that is present should be necessary in order
  to maintain reasonable accuracy. The network architecture here has no
  batch norm, no dropout. During training there are no fancy test time
  augmentations other than random flips and crops.

- It should be fast. On CIFAR-10, for example, this implementation should
  reach 90% accuracy in a few minutes on a single modern GPU. While a more
  accurate model is certainly possible, there is no pedagogical reason to
  have a model higher than ~90% accuracy.

- It should be extensible. All future defenses will extend on this training
  algorithm, so spending time to understand how it works now will be worth
  the effort.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 97% attack succes rate


## References

We build defenses to be pedagogically useful while retaining similarity to
past published defenses. This section will typically contain references
to the similar defense ideas that we are extending.

### Defenses

Since there is no defense applied, there are no important defense citations.
Any standard introduction to machine learning should cover all ideas applied
in training the neural network.

For future defenses, reading the citations in this section will be helpful before
trying to analyze the implementation and break the defense.

### Attacks

Almost all of the defenses we build here are known to be broken by attacks
that have already been published in the literature. We recommend against
reading the following papers before first trying to attack the model yourself.

After successfully breaking the defense these papers will provide ideas for
how the attack might have been broken differently.

If you get stuck, we recommend just skimming the ideas in the following papers
to draw some inspiration. Only after some time working through the defense
should you read these papers fully for the so-called "answers".
