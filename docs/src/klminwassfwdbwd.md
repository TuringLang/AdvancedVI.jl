# [`KLMinWassFwdBwd`](@id klminwassfwdbwd)

## Description

This algorithm aims to minimize the exclusive (or reverse) Kullback-Leibler (KL) divergence by running proximal gradient descent (also known as forward-backward splitting) in Wasserstein space[^DBCS2023].
(This algorithm is also sometimes referred to as "Wasserstein VI".)
Since `KLMinWassFwdBwd` is a measure-space algorithm, its use is restricted to full-rank Gaussian variational families (`FullRankGaussian`) that makes the measure-valued operations tractable.

```@docs
KLMinWassFwdBwd
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

[^DBCS2023]: Diao, M. Z., Balasubramanian, K., Chewi, S., & Salim, A. (2023). Forward-backward Gaussian variational inference via JKO in the Bures-Wasserstein space. In *International Conference on Machine Learning*. PMLR.

## [Methodology](@id klminwassfwdbwd_method)

This algorithm aims to solve the problem

```math
  \mathrm{minimize}_{q \in \mathcal{Q}}\quad \mathrm{KL}\left(q, \pi\right)
```

where $\mathcal{Q}$ is some family of distributions, often called the variational family.
Since we usually only have access to the unnormalized densities of the target distribution $\pi$, we don't have direct access to the KL divergence.
Instead, we focus on minimizing a surrogate objective, the *free energy functional*, which corresponds to the negated evidence lower bound[^JGJS1999], defined as

```math
  \mathcal{F}\left(q\right) \triangleq \mathcal{U}\left(q\right) + \mathcal{H}\left(q\right),
```

where

```math
\begin{aligned}
  \mathcal{U}\left(q\right) &= \mathbb{E}_{\theta \sim q} -\log \pi\left(\theta\right)
  &&\text{(``potential energy'')}
  \\
  \mathcal{H}\left(q\right) &= \mathbb{E}_{\theta \sim q} \log q\left(\theta\right) .
  &&\text{(``Boltzmann entropy'')}
\end{aligned}
```

For solving this problem, `KLMinWassFwdBwd` relies on proximal stochastic gradient descent (PSGD)---also known as "forward-backward splitting"---that iterates

```math
  q_{t+1} = \mathrm{JKO}_{\gamma_t \mathcal{H}}\big(
      q_{t} - \gamma_t \widehat{\nabla_{\mathrm{BW}} \mathcal{U}} (q_{t}) 
  \big) , 
```

where $$\widehat{\nabla_{\mathrm{BW}} \mathcal{U}}$$ is a stochastic estimate of the Bures-Wasserstein measure-valued gradient of $$\mathcal{U}$$, the JKO (proximal) operator is defined as

```math
\mathrm{JKO}_{\gamma_t \mathcal{H}}(\mu)
=
\argmin_{\nu \in \mathcal{Q}} \left\{ \mathcal{H}(\nu) + \frac{1}{2 \gamma_t} \mathrm{W}_2 {(\mu, \nu)}^2 \right\} ,
```

and $$\mathrm{W}_2$$ is the Wasserstein-2 distance.
When $$\mathcal{Q}$$ is set to be the Bures-Wasserstein space of $$\mathbb{R}^d$$, this algorithm is referred to as the Jordan-Kinderlehrer-Otto (JKO) scheme[^JKO1998], which was originally developed to study gradient flows under Wasserstein metrics.
Within this context, `KLMinWassFwdBwd` can be viewed as a numerical realization of the JKO scheme by restricting $$\mathcal{Q}$$ to be a tractable parametric variational family.
Specifically, Diao *et al.*[^DBCS2023] derived the JKO update for multivariate Gaussians, which is implemented by `KLMinWassFwdBwd`.
`KLMinWassFwdBwd` also exactly corresponds to the measure-space analog of [KLMinRepGradProxDescent](@ref klminrepgradproxdescent).

[^JKO1998]: Jordan, R., Kinderlehrer, D., & Otto, F. (1998). The variational formulation of the Fokker--Planck equation. *SIAM Journal on Mathematical Analysis*, 29(1).
[^JGJS1999]: Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine learning, 37, 183-233.
