# Softmax Temperature Defense

This defense trains a neural network with significantly higher "softmax
temperature" in order to flatten the loss function.


## Defense Idea

It is often easy to change the output of neural networks very quickly. By
increasing the softmax temperature the hope is that we will be able to make the
model less able to quickly change its output prediction and therefore will
be more adversarially robust.


## Training

The model is trained using almost exactly the code used to train the baseline
neural network. However during training we divide the output by a large constant
so that the neural network learns to output high-confidence predictions.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 89% attack success rate


## References

### Defenses

The softmax temperature defense idea has been developed many times, typically
unintentionally. The most canonical reference for this defense is

Papernot et al. 2016 "Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks"
https://arxiv.org/abs/1511.04508

### Attacks

Carlini & Wagner 2016 "Defensive Distillation is Not Robust to Adversarial Examples"
https://arxiv.org/abs/1607.04311