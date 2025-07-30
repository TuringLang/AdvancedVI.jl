# [Reparameterizable Variational Families](@id families)

The [RepGradELBO](@ref repgradelbo) objective assumes that the members of the variational family have a differentiable sampling path.
We provide multiple pre-packaged variational families that can be readily used.

## [The `LocationScale` Family](@id locscale)

The [location-scale](https://en.wikipedia.org/wiki/Location%E2%80%93scale_family) variational family is a family of probability distributions, where their sampling process can be represented as

```math
z \sim  q_{\lambda} \qquad\Leftrightarrow\qquad
z \stackrel{d}{=} C u + m;\quad u \sim \varphi
```

where ``C`` is the *scale*, ``m`` is the location, and ``\varphi`` is the *base distribution*.
``m`` and ``C`` form the variational parameters ``\lambda = (m, C)`` of ``q_{\lambda}``.
The location-scale family encompases many practical variational families, which can be instantiated by setting the *base distribution* of ``u`` and the structure of ``C``.

The probability density is given by

```math
  q_{\lambda}(z) = {|C|}^{-1} \varphi(C^{-1}(z - m)),
```

the covariance is given as

```math
  \mathrm{Var}\left(q_{\lambda}\right) = C \mathrm{Var}(q_{\lambda}) C^{\top}
```

and the entropy is given as

```math
  \mathbb{H}(q_{\lambda}) = \mathbb{H}(\varphi) + \log |C|,
```

where ``\mathbb{H}(\varphi)`` is the entropy of the base distribution.
Notice the ``\mathbb{H}(\varphi)`` does not depend on ``\log |C|``.
The derivative of the entropy with respect to ``\lambda`` is thus independent of the base distribution.

### API

!!! note
    
    For stable convergence, the initial `scale` needs to be sufficiently large and well-conditioned.
    Initializing `scale` to have small eigenvalues will often result in initial divergences and numerical instabilities.

```@docs
MvLocationScale
```

The following are specialized constructors for convenience:

```@docs
FullRankGaussian
MeanFieldGaussian
```

### Gaussian Variational Families

```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);

L = LowerTriangular(diagm(ones(2)));
q = FullRankGaussian(μ, L)

L = Diagonal(ones(2));
q = MeanFieldGaussian(μ, L)
```

### Student-$$t$$ Variational Families

```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);
ν = 3;

# Full-Rank 
L = LowerTriangular(diagm(ones(2)));
q = MvLocationScale(μ, L, TDist(ν))

# Mean-Field
L = Diagonal(ones(2));
q = MvLocationScale(μ, L, TDist(ν))
```

### Laplace Variational families

```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);

# Full-Rank 
L = LowerTriangular(diagm(ones(2)));
q = MvLocationScale(μ, L, Laplace())

# Mean-Field
L = Diagonal(ones(2));
q = MvLocationScale(μ, L, Laplace())
```

## The `LocationScaleLowRank` Family

In practice, `LocationScale` families with full-rank scale matrices are known to converge slowly as they require a small SGD stepsize.
Low-rank variational families can be an effective alternative[^ONS2018].
`LocationScaleLowRank` generally represent any ``d``-dimensional distribution which its sampling path can be represented as

```math
z \sim  q_{\lambda} \qquad\Leftrightarrow\qquad
z \stackrel{d}{=} D u_1 + U u_2  + m;\quad u_1, u_2 \sim \varphi
```

where ``D \in \mathbb{R}^{d \times d}`` is a diagonal matrix, ``U \in \mathbb{R}^{d \times r}`` is a dense low-rank matrix for the rank ``r > 0``, ``m \in \mathbb{R}^d`` is the location, and ``\varphi`` is the *base distribution*.
``m``, ``D``, and ``U`` form the variational parameters ``\lambda = (m, D, U)``.

The covariance of this distribution is given as

```math
  \mathrm{Var}\left(q_{\lambda}\right) = D \mathrm{Var}(\varphi) D + U \mathrm{Var}(\varphi) U^{\top}
```

and the entropy is given by the matrix determinant lemma as

```math
  \mathbb{H}(q_{\lambda}) 
  = \mathbb{H}(\varphi) + \log |\Sigma|
  = \mathbb{H}(\varphi) + 2 \log |D| + \log |I + U^{\top} D^{-2} U|,
```

where ``\mathbb{H}(\varphi)`` is the entropy of the base distribution.

```@setup lowrank
using ADTypes
using AdvancedVI
using Distributions
using LinearAlgebra
using LogDensityProblems
using Optimisers
using Plots
using ForwardDiff, ReverseDiff

struct Target{D}
    dist::D
end

function LogDensityProblems.logdensity(model::Target, θ)
    logpdf(model.dist, θ)
end

function LogDensityProblems.logdensity_and_gradient(model::Target, θ)
    return (
        LogDensityProblems.logdensity(model, θ),
        ForwardDiff.gradient(Base.Fix1(LogDensityProblems.logdensity, model), θ),
    )
end

function LogDensityProblems.dimension(model::Target)
    return length(model.dist)
end

function LogDensityProblems.capabilities(::Type{<:Target})
    return LogDensityProblems.LogDensityOrder{1}()
end

n_dims     = 30
U_true     = randn(n_dims, 3)
D_true     = Diagonal(log.(1 .+ exp.(randn(n_dims))))
Σ_true     = D_true + U_true*U_true'
Σsqrt_true = sqrt(Σ_true)
μ_true     = randn(n_dims)
model      = Target(MvNormal(μ_true, Σ_true));

d  = LogDensityProblems.dimension(model);
μ  = zeros(d);

L     = Diagonal(ones(d));
q0_mf = MeanFieldGaussian(μ, L)

L     = LowerTriangular(diagm(ones(d)));
q0_fr = FullRankGaussian(μ, L)

D     = ones(n_dims)
U     = zeros(n_dims, 3)
q0_lr = LowRankGaussian(μ, D, U)

alg = KLMinRepGradDescent(AutoReverseDiff(); optimizer=Adam(0.01))

max_iter = 10^4

function callback(; params, averaged_params, restructure, kwargs...)
    q = restructure(averaged_params)
    μ, Σ = mean(q), cov(q)
    (dist2 = sum(abs2, μ - μ_true) + tr(Σ + Σ_true - 2*sqrt(Σsqrt_true*Σ*Σsqrt_true)),)
end

_, info_fr, _ = AdvancedVI.optimize(
    alg, max_iter, model, q0_fr;
    show_progress = false,
    callback      = callback,
); 

_, info_mf, _ = AdvancedVI.optimize(
    alg, max_iter, model, q0_mf;
    show_progress = false,
    callback      = callback,
); 

_, info_lr, _ = AdvancedVI.optimize(
    alg, max_iter, model, q0_lr;
    show_progress = false,
    callback      = callback,
); 

t       = [i.iteration for i in info_fr]
dist_fr = [sqrt(i.dist2) for i in info_fr]
dist_mf = [sqrt(i.dist2) for i in info_mf]
dist_lr = [sqrt(i.dist2) for i in info_lr]
plot( t, dist_mf , label="Mean-Field Gaussian", xlabel="Iteration", ylabel="Wasserstein-2 Distance")
plot!(t, dist_fr,  label="Full-Rank Gaussian",  xlabel="Iteration", ylabel="Wasserstein-2 Distance")
plot!(t, dist_lr,  label="Low-Rank Gaussian",   xlabel="Iteration", ylabel="Wasserstein-2 Distance")
savefig("lowrank_family_wasserstein.svg")
nothing
```

Consider a 30-dimensional Gaussian with a diagonal plus low-rank covariance structure, where the true rank is 3.
Then, we can compare the convergence speed of `LowRankGaussian` versus `FullRankGaussian`:

![](lowrank_family_wasserstein.svg)

As we can see, `LowRankGaussian` converges faster than `FullRankGaussian`.
While `FullRankGaussian` can converge to the true solution since it is a more expressive variational family, `LowRankGaussian` gets there faster.

!!! info
    
    `MvLocationScaleLowRank` tend to work better with the `Optimisers.Adam` optimizer due to non-smoothness.
    Other optimisers may experience divergences.

### API

```@docs
MvLocationScaleLowRank
```

The `logpdf` of  `MvLocationScaleLowRank` has an optional argument `non_differentiable::Bool` (default: `false`).
If set as `true`, a more efficient ``O\left(r d^2\right)`` implementation is used to evaluate the density.
This, however, is not differentiable under most AD frameworks due to the use of Cholesky `lowrankupdate`.
The default value is `false`, which uses a ``O\left(d^3\right)`` implementation, is differentiable and therefore compatible with the `StickingTheLandingEntropy` estimator.

The following is a specialized constructor for convenience:

```@docs
LowRankGaussian
```

[^ONS2018]: Ong, V. M. H., Nott, D. J., & Smith, M. S. (2018). Gaussian variational approximation with a factor covariance structure. Journal of Computational and Graphical Statistics, 27(3), 465-478.
