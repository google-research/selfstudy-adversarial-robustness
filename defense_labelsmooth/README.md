# Label smoothing as a defense

This defense implements adds label smoothing to defend adversarial examples.


## Defense Idea

Adversarial examples are often said to exist because classifiers are overfit
and/or overconfident. Label smoothing is one potential way to address both of
these issues at the same time, and has been claimed to be an effective
defense many times in the past.


## Training

We train with the same training setup, however add standard label smoothing:
the loss is designed to be similar not to the one-hot [1 0 0 0 0 ...] but instead
the smoothed value [.9 .01 .01 .01 ...]. This makes the classifir more well
calibrated and therefore (hopefully) harder to attack.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 98% attack succes rate


## References

### Defenses

This defense has been suggested many times in the past. The most widely recognized
citation of this defense is the Deep Learning Book by Goodfellow et al. 2016
https://www.deeplearningbook.org/


### Attacks

A careful analyss of this problem can be found in

Ma et al. 2020 "Imbalanced Gradients: A Subtle Cause of Overestimated Adversarial Robustness."
http://arxiv.org/abs/2006.13726