# [`KLMinNaturalGradDescent`](@id klminnaturalgraddescent)

## Description

This algorithm aims to minimize the exclusive (or reverse) Kullback-Leibler (KL) divergence by running natural gradient descent.
`KLMinNaturalGradDescent` is a specific implementation of natural gradient variational inference (NGVI) also known as variational online Newton[^KR2023].
For nearly-Gaussian targets, NGVI tends to converge very quickly.
If the `ensure_posdef` option is set to `true` (this is the default configuration), then the update rule of [^LSK2020] is used, which guarantees the updated precision matrix is always positive definite.
Since `KLMinNaturalGradDescent` is a measure-space algorithm, its use is restricted to full-rank Gaussian variational families (`FullRankGaussian`) that make the updates tractable.

```@docs
KLMinNaturalGradDescent
```

The associated objective value, which is the ELBO, can be estimated through the following:

```@docs; canonical=false
estimate_objective(
    ::Random.AbstractRNG,
    ::KLMinWassFwdBwd,
    ::MvLocationScale,
    ::Any;
    ::Int,
)
```

[^KR2023]: Khan, M. E., & Rue, H. (2023). The Bayesian learning rule. *Journal of Machine Learning Research*, 24(281), 1-46.
[^LSK2020]: Lin, W., Schmidt, M., & Khan, M. E. (2020). Handling the positive-definite constraint in the Bayesian learning rule. In *International Conference on Machine Learning*. PMLR.
## [Methodology](@id klminnaturalgraddescent_method)

This algorithm aims to solve the problem

```math
  \mathrm{minimize}_{q_{\lambda} \in \mathcal{Q}}\quad \mathrm{KL}\left(q_{\lambda}, \pi\right)
```

where $\mathcal{Q}$ is some family of distributions, often called the variational family, by running stochastic gradient descent in the (Euclidean) space of parameters.
That is, for all $$q_{\lambda} \in \mathcal{Q}$$, we assume $$q_{\lambda}$$ there is a corresponding vector of parameters $$\lambda \in \Lambda$$, where the space of parameters is Euclidean such that $$\Lambda \subset \mathbb{R}^p$$.

Since we usually only have access to the unnormalized densities of the target distribution $\pi$, we don't have direct access to the KL divergence.
Instead, the ELBO maximization strategy minimizes a surrogate objective, the *negative evidence lower bound*[^JGJS1999]

```math
  \mathcal{L}\left(q\right) \triangleq \mathbb{E}_{\theta \sim q} -\log \pi\left(\theta\right) - \mathbb{H}\left(q\right),
```
which is equivalent to the KL up to an additive constant (the evidence).

Suppose we had access to the exact gradients $\nabla_{\lambda} \mathcal{L}\left(q_{\lambda}\right)$.
NGVI attempts to minimize $\mathcal{L}$ via natural gradient descent, which corresponds to iterating the mirror descent update

```math
\lambda_{t+1} =  \argmin_{\lambda \in \Lambda} {\langle \nabla_{\lambda} \mathcal{L}\left(q_{\lambda_t}\right), \lambda - \lambda_t \rangle} + \frac{1}{2 \gamma_t} \mathrm{KL}\left(q, q_{\lambda_t}\right) .
```

This turns out to be equivalent to the update

```math
\lambda_{t+1} = \lambda_{t} - \gamma_t {F(\lambda_t)}^{-1} \nabla_{\lambda} \mathcal{L}(q_{\lambda_t}) ,
```
where $F(\lambda_t)$ is the Fisher information matrix of $q_{\lambda}$.
That is, natural gradient descent can be viewed as gradient descent with an iterate-dependent preconditioning.
Furthermore, ${F(\lambda_t)}^{-1} \nabla_{\lambda} \mathcal{L}(q_{\lambda_t})$ is refered to as the *natural gradient* of the KL divergence[^A1998], hence natural gradient variational inference.
Also note that the gradient is taken over the parameters of $q_{\lambda}$.
Therefore, NGVI is parametrization dependent: for the same variational family, different parametrizations will result in different behavior.
However, the pseudo-metric $\mathrm{KL}\left(q, q_{\lambda_t}\right)$ is over measures.
Therefore, NGVI tend to behave as a measure-space algorithm, but technically speaking, not a fully measure-space algorithm.

In practice, we don't have access to $\nabla_{\lambda} \mathcal{L}\left(q_{\lambda}\right)$ apart from its unbiased estimate.
Regardless, the natural gradient descent/mirror descent updates involving the stochastic estimates have been derived for some variational families.
For instance, Gaussian variational families[^KR2023] and mixture of exponential families[^LKS2019].
As of now, we only implement the Gaussian version.

[^LKS2019]: Lin, W., Khan, M. E., & Schmidt, M. (2019). Fast and simple natural-gradient variational inference with mixture of exponential-family approximations. In *International Conference on Machine Learning*. PMLR.
[^A1998]: Amari, S. I. (1998). Natural gradient works efficiently in learning. *Neural computation*, 10(2), 251-276.
[^JGJS1999]: Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine learning, 37, 183-233.
