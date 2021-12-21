# Adversarial Training

This defense trains the model to be robust to adversarial examples by training on them.


## Defense Idea

To make a classifier good at a task we should train on exactly this task. So to
make a classifier robust to adversarial examples, we should tran on adversarial
examples. Doing this correctly requires some care, but the general idea is to
construct adversarial examples at training time, and then train the classifier to
be robust to these adversarial examples.


## Training

At each minibatch of gradient descent, we first construct one-step adversarial
examples to maximize the model loss. Then, we explicitly minimize the loss of the
model when classifying this adversarial example. 


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 48% attack succes rate


## References

### Defenses

Madry et al. 2017. "Towards Deep Learning Models Resistant to Adversarial Attacks."
http://arxiv.org/abs/1706.06083

Wong et al. 2020. "Fast is better than free: Revisiting adversarial training."
http://arxiv.org/abs/2001.03994


### Attacks

Madry et al. 2017. "Towards Deep Learning Models Resistant to Adversarial Attacks."
http://arxiv.org/abs/1706.06083
