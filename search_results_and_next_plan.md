# Search Results And Next Plan

## Comparison To Prompt Example

The prompt example is qualitatively better than our current best visual outputs.
It shows several columns that are immediately readable as strokes or edge
detectors, with simpler class-head bars. Our current outputs are quantitatively
faithful but visually noisier: many components contain multiple partial strokes
plus background texture, and the class heads are still spread across several
digits.

So the honest comparison is:

- **Fidelity:** our heavier rank-64 run is strong (`~0.95` tensor cosine,
  `~97%` decomposed accuracy).
- **Visual interpretability:** the prompt example is still better.
- **Best local family:** CP with a soft symmetry penalty and evidence-split
  variants were consistently strongest.
- **Negative result:** nonnegative-only stroke factors failed badly in this
  parameterization.

## Batch 1: Balanced Rank-32 Search

Best by combined score:

- `cp_soft_sym_l1tv`
- similarity: `0.8763`
- decomposed accuracy: `94.0%`

Notebook-selected variant:

- `evidence split sparse smooth r32`
- similarity: `0.8770`
- decomposed accuracy: `95.5%`

Artifacts:

- `figures/search/variant_search_results.csv`
- `figures/search/example_style_best_raw.png`
- `figures/search/example_style_best_denoised.png`

## Batch 2: Heavier Rank-64 Search

Best by combined score:

- `cp_soft_sym_l1tv`
- similarity: `0.9497`
- decomposed accuracy: `97.1%`

Other strong variants:

- `split_l1_tv`: similarity `0.9478`, accuracy `97.5%`
- `split_diverse`: similarity `0.9485`, accuracy `97.2%`
- `split_entropy_head`: similarity `0.9480`, accuracy `97.2%`

Artifacts:

- `figures/search_fuller/variant_search_results.csv`
- `figures/search_fuller/example_style_best_raw.png`
- `figures/search_fuller/example_style_best_denoised.png`

## Why We Are Not At The Example Yet

The current optimization still mostly rewards reconstructing the full tensor.
That admits components with high-fidelity but mixed visual support. The total
variation and L1 penalties used so far were too weak or too indirect: measured
TV barely changed across most variants. Class-head entropy also had limited
effect.

The example appears to have a stronger inductive bias toward localized stroke
features and class-sparse heads.

## Next Experiments Most Likely To Help

1. **Raw-factor smoothness instead of normalized-pattern smoothness.**
   The current TV penalty acts on normalized displayed patterns. Penalize raw
   `L`, `R`, `P`, and `N` images directly, and sweep stronger TV/Laplacian
   weights.

2. **Activation/logit distillation loss.**
   Add dataset-level faithfulness:

   ```text
   loss = tensor_loss + alpha * MSE(factor_logits(x), original_logits(x))
   ```

   This should prefer decompositions that preserve actual digit behavior, not
   arbitrary tensor mass.

3. **Hard class-head sparsity.**
   L1/entropy was not enough. Try hard-concrete gates, top-k straight-through
   gates, or proximal updates on `D` so each component only affects 1-3 digits.

4. **Localized component parameterization.**
   Learn each pattern as a small patch under a soft spatial mask, or use a
   differentiable Gaussian/window mask:

   ```text
   pattern_r = mask(center_r, width_r) * free_pixels_r
   ```

   This directly targets the prompt example's stroke-detector look.

5. **Varimax / sparse rotation after CP.**
   First fit a high-fidelity CP decomposition, then rotate the component
   dictionary toward sparse/simple factors while preserving the span. This is a
   classic route to more human-readable factors and may beat adding penalties
   during optimization.

6. **Seed-consensus averaging.**
   Match components across seeds, average recurring patterns, and display only
   stable components. This should reduce optimizer-specific noise.

7. **Sort by interpretability, not only sigma.**
   The example displays the most salient components, but our top-sigma
   components are not always the cleanest. Try ranking by:

   ```text
   sigma * locality * class_selectivity * gini
   ```

   and separately report the fidelity cost.

8. **Train the original model with stronger feature regularization.**
   Sweep Gaussian noise `std={0.3, 0.4, 0.6, 0.8}`, longer epochs, and mild
   affine/elastic augmentation. Cleaner source weights may matter more than the
   decomposition method.

9. **Edgelet/Gabor initialization.**
   Initialize factors from small oriented bars, arcs, Sobel filters, and MNIST
   stroke templates, then fine-tune. This directly tests whether the model's
   tensor can be represented by reusable human strokes.

10. **Prune and refit.**
    Fit many components, prune to the clearest/stablest subset, then refit only
    class heads or local amplitudes. This may produce a shorter and cleaner
    report figure.

