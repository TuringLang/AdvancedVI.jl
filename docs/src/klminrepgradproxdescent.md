# [`KLMinRepGradProxDescent`](@id klminrepgradproxdescent)

## Description

This algorithm is a slight variation of [`KLMinRepGradDescent`](@ref klminrepgraddescent) specialized to [location-scale families](@ref locscale).
Therefore, it also aims to minimize the exclusive (or reverse) Kullback-Leibler (KL) divergence over the space of parameters.
But instead, it uses stochastic proximal gradient descent with the [proximal operator](@ref proximalocationscaleentropy) of the entropy of location-scale variational families as discussed in: [^D2020][^KMG2024][^DGG2023].
The remainder of the section will only discuss details specific to `KLMinRepGradProxDescent`.
Thus, for general usage and additional details, please refer to the docs of `KLMinRepGradDescent` instead.

```@docs
KLMinRepGradProxDescent
```

The associated objective value can be estimated through the following:
```@docs
estimate_objective(
    ::Random.AbstractRNG,
    ::KLMinRepGradProxDescent,
    ::Any,
    ::Any;
    ::Int,
    ::AbstractEntropyEstimator,
)
```

## Methodology

Recall that [KLMinRepGradDescent](@ref klminrepgraddescent) maximizes the ELBO.
Now, the ELBO can be re-written as follows:

```math
  \mathrm{ELBO}\left(q\right) \triangleq \mathcal{E}\left(q\right) + \mathbb{H}\left(q\right),
```

where

```math
  \mathcal{E}\left(q\right) = \mathbb{E}_{\theta \sim q} \log \pi\left(\theta\right)
```

is often referred to as the *negative energy functional*.
`KLMinRepGradProxDescent` attempts to address the fact that minimizing the whole ELBO can be unstable due to non-smoothness of $$\mathbb{H}\left(q\right)$$[^D2020].
For this, `KLMinRepGradProxDescent` relies on proximal stochastic gradient descent, where the problematic term $$\mathbb{H}\left(q\right)$$ is separately handled via a *proximal operator*.
Specifically, `KLMinRepGradProxDescent` first estimates the gradient of the energy $$\mathcal{E}\left(q\right)$$ only via the reparameterization gradient estimator.
Let us denote this as $$\widehat{\nabla_{\lambda} \mathcal{E}}\left(q_{\lambda}\right)$$.
Then `KLMinRepGradProxDescent` iterates the step

```math
  \lambda_{t+1} = \mathrm{prox}_{-\gamma_t \mathbb{H}}\big( 
      \lambda_{t} + \gamma_t \widehat{\nabla_{\lambda} \mathcal{E}}(q_{\lambda_t})
  \big) , 
```

where

```math
\mathrm{prox}_{h}(\lambda_t) 
= \argmin_{\lambda \in \Lambda}\left\{ 
    h(\lambda) + {\lVert \lambda - \lambda_t \rVert}_2^2 
\right\}
```

is a proximal operator for the entropy.
As long as $$\mathrm{prox}_{-\gamma_t \mathbb{H}}$$ can be evaluated efficiently, this scheme can side-step the fact that $$\mathbb{H}(\lambda)$$ is difficult to deal with via gradient descent.
For location-scale families, it turns out the proximal operator of the entropy can be operated efficiently[^D2020], which is implemented as [`ProximalLocationScaleEntropy`](@ref proximalocationscaleentropy).
This has been empirically shown to be more robust[^D2020][^KMG2024].

[^D2020]: Domke, J. (2020). Provable smoothness guarantees for black-box variational inference. In *International Conference on Machine Learning*.
[^KMG2024]: Kim, K., Ma, Y., & Gardner, J. (2024). Linear Convergence of Black-Box Variational Inference: Should We Stick the Landing?. In International Conference on Artificial Intelligence and Statistics (pp. 235-243). PMLR.
[^DGG2023]: Domke, J., Gower, R., & Garrigos, G. (2023). Provable convergence guarantees for black-box variational inference. Advances in neural information processing systems, 36, 66289-66327.
