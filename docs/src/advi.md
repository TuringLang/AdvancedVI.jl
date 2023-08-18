
# [Automatic Differentiation Variational Inference](@id advi)

## Introduction

The automatic differentiation variational inference (ADVI; Kucukelbir *et al.* 2017) objective is a method for estimating the evidence lower bound between a target posterior distribution ``\pi`` and a variational approximation ``q_{\phi,\lambda}``.
By maximizing ADVI objective, it is equivalent to solving the problem

```math
  \mathrm{minimize}_{\lambda \in \Lambda}\quad \mathrm{KL}\left(q_{\phi,\lambda}, \pi\right).
```

The key aspects of the ADVI objective are the followings:
1. The use of the reparameterization gradient estimator
2. Automatically match the support of the target posterior through "bijectors."

Thanks to Item 2, the user is free to choose any unconstrained variational family, for which
bijectors will automatically match the potentially constrained support of the target.

In particular, ADVI implicitly forms a variational approximation ``q_{\phi,\lambda}``
from a reparameterizable distribution ``q_{\lambda}`` and a bijector ``\phi`` such that
```math
z \sim  q_{\phi,\lambda} \qquad\Leftrightarrow\qquad
z \stackrel{d}{=} \phi^{-1}\left(\eta\right);\quad \eta \sim q_{\lambda} 
```
ADVI provides a principled way to compute the evidence lower bound for ``q_{\phi,\lambda}``.

That is,

```math
\begin{aligned}
\mathrm{ADVI}\left(\lambda\right)
&\triangleq
\mathbb{E}_{\eta \sim q_{\lambda}}\left[
  \log \pi\left( \phi^{-1}\left( \eta \right) \right)
\right]
+ \mathbb{H}\left(q_{\lambda}\right)
+ \log \lvert J_{\phi^{-1}}\left(\eta\right) \rvert \\
&=
\mathbb{E}_{\eta \sim q_{\lambda}}\left[
  \log \pi\left( \phi^{-1}\left( \eta \right) \right)
\right]
+
\mathbb{E}_{\eta \sim q_{\lambda}}\left[
  - \log q_{\lambda}\left( \eta \right) \lvert J_{\phi}\left(\eta\right) \rvert
\right] \\
&=
\mathbb{E}_{z \sim q_{\phi,\lambda}}\left[ \log \pi\left(z\right) \right]
+
\mathbb{H}\left(q_{\phi,\lambda}\right)
\end{aligned}
```

The idea of using the reparameterization gradient estimator for variational inference was first 
coined by Titsias and Lázaro-Gredilla (2014).
Bijectors were generalized by Dillon *et al.* (2017) and later implemented in Julia by
Fjelde *et al.* (2017).

## The `ADVI` Objective

```@docs
ADVI
```

## The `StickingTheLanding` Control Variate

The STL control variate was proposed by Roeder *et al.* (2017).
By slightly modifying the differentiation path, it implicitly forms a control variate of the form of
```math
\begin{aligned}
  \mathrm{CV}_{\mathrm{STL}}\left(z\right) 
  &\triangleq 
  \nabla_{\lambda} \mathbb{H}\left(q_{\lambda}\right) + \nabla_{\lambda} \log q_{\nu}\left(z_{\lambda}\left(u\right)\right) \\
  &=
  -\nabla_{\lambda} \mathbb{E}_{z \sim q_{\nu}} \log q_{\nu}\left(z_{\lambda}\left(u\right)\right) + \nabla_{\lambda} \log q_{\nu}\left(z_{\lambda}\left(u\right)\right)
\end{aligned}
```
where ``\nu = \lambda`` is set to avoid differentiating through the density of ``q_{\lambda}``.
We can see that this vector-valued function has a mean of zero and is therefore a valid control variate.
 
Adding this to the closed-form entropy ELBO estimator yields the STL estimator:
```math
\begin{aligned}
  \widehat{\nabla \mathrm{ELBO}}_{\mathrm{STL}}\left(\lambda\right)
    &\triangleq \mathbb{E}_{u \sim \varphi}\left[ 
	  \nabla_{\lambda} \log \pi \left(z_{\lambda}\left(u\right)\right) 
	  - 
	  \nabla_{\lambda} \log q_{\nu} \left(z_{\lambda}\left(u\right)\right)
	\right] 
	\\
    &= 
	\mathbb{E}\left[ \nabla_{\lambda} \log \pi\left(z_{\lambda}\left(u\right)\right) \right] 
    + 
	\nabla_{\lambda} \mathbb{H}\left(q_{\lambda}\right) 
	- 
	\mathrm{CV}_{\mathrm{STL}}\left(z\right)
	\\
    &= 
	\widehat{\nabla \mathrm{ELBO}}\left(\lambda\right)
    - 
	\mathrm{CV}_{\mathrm{STL}}\left(z\right),
\end{aligned}
```
which has the same expectation as the original ADVI estimator, but lower variance when ``\pi \approx q_{\lambda}``, and higher variance when ``\pi \not\approx q_{\lambda}``.
The conditions for which the STL estimator results in lower variance is still an active subject for research.

The main downside of the STL estimator is that it needs to evaluate and differentiate the log density of ``q_{\lambda}`` in every iteration.
Depending on the variational family, this might be computationally inefficient or even numerically unstable.
For example, if ``q_{\lambda}`` is a Gaussian with a full-rank covariance, a back-substitution must be performed at every step, making the per-iteration complexity ``\mathcal{O}(d^3)`` and reducing numerical stability.


The STL control variate can be used by changing the entropy estimator using the following object:
```@docs
StickingTheLandingEntropy
```

```@setup stl
using LogDensityProblems
using SimpleUnPack
using PDMats
using Bijectors
using LinearAlgebra
using Plots

using Optimisers
using ADTypes, ForwardDiff
import AdvancedVI as AVI

struct NormalLogNormal{MX,SX,MY,SY}
    μ_x::MX
    σ_x::SX
    μ_y::MY
    Σ_y::SY
end

function LogDensityProblems.logdensity(model::NormalLogNormal, θ)
    @unpack μ_x, σ_x, μ_y, Σ_y = model
    logpdf(LogNormal(μ_x, σ_x), θ[1]) + logpdf(MvNormal(μ_y, Σ_y), θ[2:end])
end

function LogDensityProblems.dimension(model::NormalLogNormal)
    length(model.μ_y) + 1
end

function LogDensityProblems.capabilities(::Type{<:NormalLogNormal})
    LogDensityProblems.LogDensityOrder{0}()
end

n_dims = 10
μ_x    = randn()
σ_x    = exp.(randn())
μ_y    = randn(n_dims)
σ_y    = exp.(randn(n_dims))
model  = NormalLogNormal(μ_x, σ_x, μ_y, PDMats.PDiagMat(σ_y.^2));

d  = LogDensityProblems.dimension(model);
μ  = randn(d);
L  = Diagonal(ones(d));
q0 = AVI.VIMeanFieldGaussian(μ, L)

model  = NormalLogNormal(μ_x, σ_x, μ_y, PDMats.PDiagMat(σ_y.^2));

function Bijectors.bijector(model::NormalLogNormal)
    @unpack μ_x, σ_x, μ_y, Σ_y = model
    Bijectors.Stacked(
        Bijectors.bijector.([LogNormal(μ_x, σ_x), MvNormal(μ_y, Σ_y)]),
        [1:1, 2:1+length(μ_y)])
end
```

Let us come back to the example in [Getting Started](@ref getting_started), where a `LogDensityProblem` is given as `model`.
In this example, the true posterior is contained within the variational family.
This setting is known as "perfect variational family specification."
In this case, the STL estimator is able to converge exponentially fast to the true solution.

Recall that the original ADVI objective with a closed-form entropy (CFE) is given as follows:
```@example stl
n_montecarlo = 1;
b            = Bijectors.bijector(model);
b⁻¹          = inverse(b)

cfe = AVI.ADVI(model, n_montecarlo; invbij = b⁻¹)
```
The STL estimator can instead be created as follows:
```@example stl
stl = AVI.ADVI(model, n_montecarlo; entropy = AVI.StickingTheLandingEntropy(), invbij = b⁻¹);
```

```@setup stl
n_max_iter = 10^4

_, stats_cfe, _ = AVI.optimize(
    cfe,
    q0,
    n_max_iter;
	show_progress = false,
    adbackend     = AutoForwardDiff(),
    optimizer     = Optimisers.Adam(1e-3)
); 

_, stats_stl, _ = AVI.optimize(
    stl,
    q0,
    n_max_iter;
	show_progress = false,
    adbackend     = AutoForwardDiff(),
    optimizer     = Optimisers.Adam(1e-3)
); 

t     = [stat.iteration  for stat ∈ stats_cfe]
y_cfe = [stat.elbo       for stat ∈ stats_cfe]
y_stl = [stat.elbo       for stat ∈ stats_stl]
plot( t, y_cfe, label="ADVI CFE", xlabel="Iteration", ylabel="ELBO")
plot!(t, y_stl, label="ADVI STL", xlabel="Iteration", ylabel="ELBO")
savefig("advi_stl_elbo.svg")
nothing
```
![](advi_stl_elbo.svg)

We can see that the noise of the STL estimator becomes smaller as VI converges.
However, the speed of convergence may not always be significantly different.


## References
1. Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. Journal of machine learning research.
2. Titsias, M., & Lázaro-Gredilla, M. (2014, June). Doubly stochastic variational Bayes for non-conjugate inference. In International conference on machine learning (pp. 1971-1979). PMLR.
3. Dillon, J. V., Langmore, I., Tran, D., Brevdo, E., Vasudevan, S., Moore, D., ... & Saurous, R. A. (2017). Tensorflow distributions. arXiv preprint arXiv:1711.10604.
4. Fjelde, T. E., Xu, K., Tarek, M., Yalburgi, S., & Ge, H. (2020, February). Bijectors. jl: Flexible transformations for probability distributions. In Symposium on Advances in Approximate Bayesian Inference (pp. 1-17). PMLR.
5. Roeder, G., Wu, Y., & Duvenaud, D. K. (2017). Sticking the landing: Simple, lower-variance gradient estimators for variational inference. Advances in Neural Information Processing Systems, 30.


