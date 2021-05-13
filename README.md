# Self-study course in evaluating adversarial robustness

This repository contains a collection of defenses aimed at researchers who wish to learn how to properly evaluate the robustness of adversarial example defenses.

While there is a vast literature of published techniques that help to attack adversarial example defenses, few researchers have practical experience actually running these. This project is designed to give researchers that experience, so that when they develop their own defenses, they can perform a thorough evaluation.

The primary purpose of these defenses is therefore to be pedagogically useful. We aim to make our defenses as simple as possible---even sacrificing accuracy or robustness if we can make a defense simpler.

This simplicity-above-all-else has one cost: it possible for one to be able to break all of our defenses, but still not properly evaluate the robustness of their own (often more complicated codebase) codebase. Researchers who aim to build strong defenses would be well served by first creating a defense as simple as we have here and then analyzing it.

This is currently a preliminary code release. We expect that we will make changes to the defenses, training algorithms, and possibly framework, as we receive feedback. We have no immediate timeline for when it will become stable.

A whitepaper describing this project will be coming in the future. More details on getting started to use this project are available in the following three documents.

* [Installation and getting started](docs/getting_started.md)
* [How to contribute](docs/contributing.md)
* [Training documentation](training/README.md)

## List of all defenses

The list below goes (roughly) from the easiest to easier to harder defenses. The first defenses require very little overall knowledge, and by the end we hope to cover all modern defense techniques. However, feel free to study them in any convenient order.

1. Baseline ([`defense_baseline`](defense_baseline/))
   * This is naively trained model without any hardening against adversarial examples.
1. Bluring ([`defense_blur`](defense_blur/))
   * This model blurs its input images in an attempt to remove adversarial perturbation.
1. Softmax temperature ([`defense_temperature`](defense_temperature/))
   * This model trains a neural network with increased softmax temperature.
1. Ensemble of binary classifiers ([`defense_mergebinary`](defense_mergebinary/))
   * The model works by merging predictions of independent per-class binary classfiers.
1. Label smoothing ([`defense_labelsmooth`](defense_labelsmooth/))
   * This model is trained with a label smoothing objective.
1. Jump ([`defense_jump`](defense_jump/))
   * This model applies a "jump" activation function instead of ReLU.
1. Majority vote ([`defense_majority`](defense_majority/))
   * This model takes a best-of-three majority vote among separate classifiers.
1. Discretization ([`defense_discretize`](defense_discretize/))
   * This model encodes the input into a more sophisticated discretization of the input.
1. Random neuron perturbations ([`defense_randomneuron`](defense_randomneuron/))
   * This model adds test-time randomness to the activations of a neural network.
1. Transform ([`defense_transform`](defense_transform/))
   * This model randomly modifies the input image before classifying it.
1. Injection ([`defense_injection`](defense_injection/))
   * This model injects backdoors into a model so that inputs can be fingerprinted.
1. K-nearest neighbours ([`defense_knn`](defense_knn/))
   * This model embeds inputs into representation space and then uses a nearest neighbor classifier.
1. Simple adversarial training ([`defense_advtrain`](defense_advtrain/))
   * This model trains on adversarial examples to become robust to them.

## Repository structure

* [armory_compat/](armory_compat/) - directory with experimental support for [Armory framework](https://github.com/twosixlabs/armory).
* [checkpoints/](checkpoints/) - directory where pre-trained checkpoints will be saved.
* [common/](common/) - common code which will be used by various other modules, this includes dataset code and model code.
* [docs/](docs/) - location of additional documentaion files
* [defense_NAME](defense_NAME) - code for the defense with name [NAME](NAME). Directory with each defense should have following structure:
    * [defense_NAME/README.md](defense_NAME/README.md) - a description of what this defense does. Read [defense_baseline](defense_baseline/README.md) first to understand their structure.
    * [defense_NAME/model.py](defense_NAME/model.py) - code of the defense model.
    * [defense_NAME/attack_{type}.py](defense_NAME/attack_{type}.py) - placeholder for the attack.
    * [defense_NAME/task_definition.py](defense_NAME/task_definition.py) - task definition for the evaluation code.
* [evaluate.py](evaluate.py) - helper script which performs evaluation and grading of tasks
