# [`FisherMinBatchMatch`](@id fisherminbatchmatch)

## Description

This algorithm, known as batch-and-match (BaM) aims to minimize the covariance-weighted 2nd-order Fisher divergence by running a proximal point-type method[^CMPMGBS24].
On certain low-dimensional problems, BaM can converge very quickly without any tuning.
Since `FisherMinBatchMatch` is a measure-space algorithm, its use is restricted to full-rank Gaussian variational families (`FullRankGaussian`) that make the measure-valued operations tractable.

```@docs
FisherMinBatchMatch
```

The associated objective value can be estimated through the following:

```@docs; canonical=false
estimate_objective(
    ::Random.AbstractRNG,
    ::KLMinWassFwdBwd,
    ::MvLocationScale,
    ::Any;
    ::Int,
)
```

[^CMPMGBS24]: Cai, D., Modi, C., Pillaud-Vivien, L., Margossian, C. C., Gower, R. M., Blei, D. M., & Saul, L. K. (2024). Batch and match: black-box variational inference with a score-based divergence. In *Proceedings of the International Conference on Machine Learning*.
## [Methodology](@id fisherminbatchmatch_method)

This algorithm aims to solve the problem

```math
  \mathrm{minimize}_{q \in \mathcal{Q}}\quad \mathrm{F}_{\mathrm{cov}}(q, \pi),
```

where $\mathcal{Q}$ is some family of distributions, often called the variational family, and $\mathrm{F}_{\mathrm{cov}}$ is a divergence defined as

```math
\mathrm{F}_{\mathrm{cov}}(q, \pi) = \mathbb{E}_{z \sim q} {\left\lVert \nabla \log \frac{q}{\pi} (z) \right\rVert}_{\mathrm{Cov}(q)}^2 ,
```

where ${\lVert x \rVert}_{A}^2 = x^{\top} A x $ is a weighted norm.
$\mathrm{F}_{\mathrm{cov}}$ can be viewed as a variant of the canonical 2nd-order Fisher divergence defined as

```math
\mathrm{F}_{2}(q, \pi) = \sqrt{ \mathbb{E}_{z \sim q} {\left\lVert \nabla \log \frac{q}{\pi} (z) \right\rVert}^2 }.
```

The use of the weighted norm ${\lVert \cdot \rVert}_{\mathrm{Cov}(q)}^2$ facilitates the use of a proximal point-type method for minimizing $\mathrm{F}_{2}(q, \pi)$.
In particular, BaM iterates the update

```math
  q_{t+1} = \argmin_{q \in \mathcal{Q}} \left\{ \mathrm{F}_{\mathrm{cov}}(q, \pi) + \frac{2}{\lambda_t} \mathrm{KL}\left(q_t, q\right) \right\} .
```

Since $\mathrm{F}(q, \pi)$ is intractable, it is replaced with a Monte Carlo approximation with a number of samples `n_samples`.
Furthermore, by restricting $\mathcal{Q}$ to a Gaussian variational family, the update rule admits a closed form solution[^CMPMGBS24].
Notice that the update does not involve the parameterization of $q_t$, which makes `FisherMinBatchMatch` a measure-space algorithm.

Historically, the idea of using a proximal point-type update for minimizing a Fisher divergence-like objective was initially coined as Gaussian score matching[^MGMYBS23].
BaM can be viewed as a successor to this algorithm.

[^MGMYBS23]: Modi, C., Gower, R., Margossian, C., Yao, Y., Blei, D., & Saul, L. (2023). Variational inference with Gaussian score matching. In *Advances in Neural Information Processing Systems*, 36.
