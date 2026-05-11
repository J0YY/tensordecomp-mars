(Just for my refernce) - MARS V Applicant task
Decomposing Weights into Human Concepts 
Thomas Dooms & Ward Gauderis

This task provides hands-on experience of weight-based decompositions for interpretability. It’s meant for us to assess your skills, but also for yourself to see whether you like this type of research. The end goal is a concise report with screenshots of experiments and thoughts/conclusions. Linking to the code is encouraged although we likely won’t read it.

Setup (~1 hour) 
Go to this repo and read the first two tutorials (introduction & images). It describes how to decompose MNIST models into interpretable components along with code for training and setup. Take some time to understand the motivation and the math as you’ll need it for the implementation of the task. Without it, you’re flying blind.

Experimenting (1-2 hours)
The primary issue with an eigendecomposition is the orthogonality of the eigenvectors; there might exist overlapping structures in input space that behave differently. Eigenvectors won’t properly capture these, yielding ugly “superposed” features. We might hope that a 6 decomposes into a few orthogonal edge-detectors but that generally doesn’t happen. Luckily, eigendecompositions aren’t the only way to decompose a matrix or tensor.

The goal of this part is to implement a tensor decomposition that is more in line with our desiderata. This is where you try to find what kinds of priors/structure would work well. We’ve written a skeleton with the essentials for you here. There’s no right answer; we’re mainly interested in your thought process and proposed experiments.

Here’s what we ended up with after about 1 hour. The first column shows that a vertical slice contributes positively to digits 1 and 4 while a horizontal bar negatively contributes. There are various edge detectors for which make sense at a first glance.

Sparsifying the weights is a good way to find interpretable components but not the only way. It’s up to you to find how to make the decomposition interpretable.
