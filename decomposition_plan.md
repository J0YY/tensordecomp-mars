# Decomposition Notebook Plan

## Goal

Turn `0_decomposition.ipynb` into a compact experiment report for the MARS V task:
compare tensor decompositions of the MNIST bilinear interaction tensor, looking for
components that are short, faithful to the original model, stable across seeds, and
human-readable.

The notebook should not only show a good-looking decomposition. It should show the
search process: which priors helped, which failed, and how we measured the tradeoff
between reconstruction quality and interpretability.

## Baseline

Start from the provided `image.sparse.Model` CP-like decomposition:

```text
B[c, i, j] ~= sum_r L[i, r] R[j, r] D[c, r]
```

Record:

- original model accuracy
- sparse approximation accuracy
- tensor cosine similarity
- component norm spectrum
- top component visualizations: `L + R`, `L - R`, class logits

This is the control condition for all variants.

## Evaluation Harness

Add reusable functions for:

- computing the original interaction tensor from the trained bilinear model
- cosine similarity between original and approximate tensors
- test accuracy of each factorized model
- pattern sparsity / Gini score
- spatial locality / patch concentration score
- class selectivity of output weights
- component ablation and keep-only faithfulness curves
- seed stability by matching learned components across runs
- standardized component plots
- activation galleries: train examples that most activate a component

The table of metrics matters as much as the visuals.

## Variants To Try

### 1. Provided Sparse Baseline

Use the original `Sparse.from_config(rank=64)` optimization as a familiar starting
point. This is expected to reconstruct reasonably well but can produce somewhat
superposed or visually noisy components.

### 2. Symmetric Factors

Force the left and right input factors to share one vector:

```text
B[c, i, j] ~= sum_r V[i, r] V[j, r] D[c, r]
```

This is less expressive than the baseline but better matched to the symmetrized
interaction tensor. The hypothesis is that it gives cleaner, more eigendecomposition-
like features while still sharing components across classes.

### 3. Sparse + Smooth Symmetric Factors

Add image-domain priors:

- L1 sparsity on input patterns
- total variation on `28 x 28` patterns
- class sparsity on `D`

The hypothesis is that a small reconstruction penalty buys much more readable
stroke-like components.

### 4. Signed Evidence Split

Learn positive and negative evidence patterns directly:

```text
activation_r(x) = (P_r^T x)^2 - (N_r^T x)^2
logit_c += D[c, r] * activation_r(x)
```

This makes the notebook's `L + R` / `L - R` interpretation the actual
parameterization, not a derived visualization.

### 5. Nonnegative Strokes

Parameterize image patterns with `softplus` while keeping class weights signed.
MNIST strokes are mostly additive dark-pixel evidence; the output direction should
carry whether a stroke helps or hurts a digit.

### 6. Eigenvector-Seeded Dictionary

Initialize global components from the top per-class eigenvectors of the trained
model. This tests whether a shared tensor dictionary can compress and merge the
class-local eigenfeatures from tutorial 1.

### 7. Multi-Seed Consensus

Run the strongest one or two variants across several seeds, align components by
cosine similarity, and report the recurring components. A recurring component is
more compelling than a single nice visualization.

## Standout Visuals

The README and notebook should include:

- baseline vs best variant top components
- metric table across variants
- ablation / keep-only curves
- activation galleries for named components
- seed-consensus grid
- a concise conclusion panel: "what worked, what failed, what I would try next"

## Expected Conclusion Shape

The strongest expected result is not perfect reconstruction. It is a measured
tradeoff:

> Plain tensor reconstruction captures shared features but often leaves visual
> superposition. Symmetric, sparse, smooth, and class-selective priors slightly
> reduce reconstruction quality while improving locality, stability, and human
> readability. The best components behave like reusable digit strokes rather than
> per-class eigenvectors.

