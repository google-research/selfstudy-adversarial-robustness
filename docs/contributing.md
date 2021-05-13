# How to contribute

## Adding New Defenses

Do you have ideas for defenses that we haven't yet implemented? We welcome contributions.

Defenses should be designed to be pedagogically interesting.

1. The lessons learned from any case study defense should apply to more than one research paper.

2. Implementations should be as simple as possible. Sacrificing accuracy for simplicity is desirable.

A word of warning, however. We are going to be cautious at accepting all new defenses in order to avoid causing bloat. More defenses is not strictly better, and defenses that don't teach a useful lesson can just waste everyone's time. When submitting a PR, it would be helpful to provide a justification why this defense requires a sufficiently different attack idea than the other defenses. If you're not sure, start with an issue and propose the idea.

### Specific defense ideas

There are still a few defense categories we know that we're missing. The reason we haven't added them yet is that we haven't found a way to implement it that's not overly complicated. We would gladly accept PRs with nice implementations of these defenses.

- *Learned detector*. Generat adversarial examples on a baseline classifier, train a detection algorithm to separate them from clean examples, and then deploy this detector. All of our solutions that tried this and weren't trivially broken required ugly training code.

- *Manifold projection*. Compute a manifold of the latent space of images, and then project images onto the manifold before classification. While the idea here is simple, the devil is in the details and most cifar-10 implementations require training GANs or other complicated generative models which are not nice to analyze.

- *Generative classifiers*. Try to reconstruct the input with N generative models and return as the label the generator that best reconstructs the input. Again, this defense requires training generators (on cifar-10) and this generally is ugly and difficult. A simple implementation would be great.

## Removing Existing Defenses

Do you have thoughts on defenses that we have implemented that are *not* pedagogically useful and we should remove? Are there three "linearly dependent" defenses where two of the three would suffice? We likewise welcome suggestions to remove the amount of time spent not learning new concepts.

## Simplifying Existing Defenses

Less dramatic than the above, is there a defense that can be made simpler and still retain respectable accuracy (~70% or so?) while maintaining its pedagogical purpose? If you can do this, please submit a PR.

## Contributing attack results

In each defense's README we give the best attack results we have been able to achieve. If you are able to do better than the best we've done, please submit a PR to update the number. (Note: we have not yet put in these numbers. They should be in soon.)

We don't want this to become a "leaderboard" in any way---this kind of competitive thought process often harms one's ability to learn by changing the dynamic from "how do I understand this deeply" to "I'll do whatever I need to just make this number better". So, if you are able to do better, please just change the number to be a lower value. If you want to prepare a separate document describing how the lower number was obtained, go ahead! It would be helpful for people to read though afterwards.

