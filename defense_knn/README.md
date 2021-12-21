# K nearest neighbor classification

This defense classifies adversarial examples with a k-nearest neighbor classifier.


## Defense Idea

Instaed of using a neural network to classify inputs end-to-end, this defense
first constructs an embedding of a given test input with a neural network, but
then uses a k nearest neighbor classifier for final classification. Specifically,
the entire training dataset is embedded (with the same neural network embedding
function) and then the model outputs the label as determined by a majority vote
of the eight nearest neighbors in embedding space.


## Training

The embedding function is trained with a SimCLR-style training objective. We
disregard the labels, and only train the model so that augmented versions of the
same input map onto similar locations in the embedding space. This encourages
good representation learning, which can be used for a k nearest neighbor classifier.


## Objectives

Our best attack results are as follows, when evaluated on the first 100 images
in the CIFAR-10 test set using the provided pretrained model..
- l_infinity distortion of 4/255: 62% attack succes rate


## References

### Defenses

There are a number of defenses that apply this style of classifier, but they are
not typically used as adversarial example defenses. This paper is the most related:

Chen et al. 2020. "A Simple Framework for Contrastive Learning of Visual Representations"
https://arxiv.org/abs/2002.05709

This paper explicitly defines a defense with a kNN:

Sitawarin et al. 2019 "Defending Against Adversarial Examples with K-Nearest Neighbor."
http://arxiv.org/abs/1906.09525

### Attacks

There are a number of papers developing attacks on kNN classifiers, but this is
an understudied area of research

Sitawarin et al. 2020 "Adversarial Examples for $k$-Nearest Neighbor Classifiers Based on Higher-Order Voronoi Diagrams."
http://arxiv.org/abs/2011.09719