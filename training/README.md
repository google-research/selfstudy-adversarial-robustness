# Training algorithms

This directory implements the training algorithms for each defense.
While we describe how each of the individual defenses' are trained in their corresponding
README files, here we describe the general training code that is shared amongst all defenses.

The baseline training loop is provided in [train_baseline.py](train_baseline.py). It loads
the data from the data loader, creates a TrainLoop, and then calls into it.

The TrainLoop is a standard training loop that uses momentum SGD to train an all convolutional
neural network to ~95% accuracy. The model uses entirely 3x3 convolutions followed by average
pooling every other layer. The final prediction is returned by using a 1x1 convolution that
gives ten channels out, and then a final global average pool across the spatial dimensions.

We train the models with standard cross entropy loss, a batch size of 64, a learning rate of
0.03, and a weight decay of 1e-4. These constants aren't all that sensitive, but we found
them to work reasonably well.

Before training on any individual image we augment it with a random shift in both the
horizontal and vertical dimensions, and then optionally flip the image across the horizontal
axis.

Finally, in order to gain a few more percentage points in accuracy, we keep an exponential
moving average of model parameters and save these EMA weights instead of the original weights.
This makes it possible to use a simpler learning rate schedule while still converging to
a high quality solution.

If the training loop ever detects the loss is NaN, it early aborts. Most defenses are stable,
but some will diverge in rare situations. If this happens re-running the training loop will
solve the problem.

## Converting to PyTorch

The above code is implemented in TensorFlow 2. After training models, it is possible to
convert them to load in PyTorch by running the provided  [convert_pytorch.py](/convert_pytorch.py) script.