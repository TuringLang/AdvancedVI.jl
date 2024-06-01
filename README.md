
# AdvancedVI.jl
[AdvancedVI](https://github.com/TuringLang/AdvancedVI.jl) provides implementations of variational inference (VI) algorithms.
VI algorithms perform scalable and computationally efficient Bayesian inference at the cost of asymptotic exactness.
`AdvancedVI` is part of the [Turing](https://turinglang.org/stable/) probabilistic programming ecosystem.
The purpose of this package is to provide a common accessible interface for various VI algorithms and utilities so that other packages, e.g. `Turing`, only need to write a light wrapper for integration. 
For example, integrating `Turing` with  `AdvancedVI.ADVI` only involves converting a `Turing.Model` into a [`LogDensityProblem`](https://github.com/tpapp/LogDensityProblems.jl) and extracting a corresponding `Bijectors.bijector`.

## Examples

`AdvancedVI` expects a `LogDensityProblem`.
For example, for the normal-log-normal model:

$$
\begin{aligned}
x &\sim \mathrm{LogNormal}\left(\mu_x, \sigma_x^2\right) \\
y &\sim \mathcal{N}\left(\mu_y, \sigma_y^2\right),
\end{aligned}
$$

a `LogDensityProblem` can be implemented as 
```julia
using LogDensityProblems
using SimpleUnPack

struct NormalLogNormal{MX,SX,MY,SY}
    μ_x::MX
    σ_x::SX
    μ_y::MY
    Σ_y::SY
end

function LogDensityProblems.logdensity(model::NormalLogNormal, θ)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    logpdf(LogNormal(μ_x, σ_x), θ[1]) + logpdf(MvNormal(μ_y, Σ_y), θ[2:end])
end

function LogDensityProblems.dimension(model::NormalLogNormal)
    length(model.μ_y) + 1
end

function LogDensityProblems.capabilities(::Type{<:NormalLogNormal})
    LogDensityProblems.LogDensityOrder{0}()
end
```

Since the support of `x` is constrained to be positive, and inference is best done in the unconstrained Euclidean space, we need to use a *bijector* to match support.
This corresponds to the automatic differentiation variational inference (ADVI) formulation[^KTRGB2017].
```julia
using Bijectors

function Bijectors.bijector(model::NormalLogNormal)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    Bijectors.Stacked(
        Bijectors.bijector.([LogNormal(μ_x, σ_x), MvNormal(μ_y, Σ_y)]),
        [1:1, 2:1+length(μ_y)])
end
```

A simpler approach is to use `Turing`, where a `Turing.Model` can be automatically be converted into a `LogDensityProblem` and a corresponding `bijector` is automatically generated.

Let us instantiate a random normal-log-normal model.
```julia
using LinearAlgebra

n_dims = 10
μ_x    = randn()
σ_x    = exp.(randn())
μ_y    = randn(n_dims)
σ_y    = exp.(randn(n_dims))
model  = NormalLogNormal(μ_x, σ_x, μ_y, Diagonal(σ_y.^2))
```

We can perform VI with stochastic gradient descent (SGD) using reparameterization gradient estimates of the ELBO[^TL2014][^RMW2014][^KW2014] as follows:
```julia
using Optimisers
using ADTypes, ForwardDiff
using AdvancedVI

# ELBO objective with the reparameterization gradient
n_montecarlo = 10
elbo         = AdvancedVI.RepGradELBO(n_montecarlo)

# Mean-field Gaussian variational family
d = LogDensityProblems.dimension(model)
μ = zeros(d)
L = Diagonal(ones(d))
q = AdvancedVI.MeanFieldGaussian(μ, L)

# Match the support of `model` by applying the bijector
b       = Bijectors.bijector(model)
binv    = inverse(b)
q_trans = Bijectors.TransformedDistribution(q, binv)


# Run inference
max_iter = 10^3
q, stats, _ = AdvancedVI.optimize(
    model,
    elbo,
    q_trans,
    max_iter;
    adbackend = ADTypes.AutoForwardDiff(),
    optimizer = Optimisers.Adam(1e-3)
)

# Evaluate final ELBO with 10^3 Monte Carlo samples
estimate_objective(elbo, q, model; n_samples=10^4)
```

For more examples and details, please refer to the documentation.

## References
[^TL2014]: Titsias, M., & Lázaro-Gredilla, M. (2014, June). Doubly stochastic variational Bayes for non-conjugate inference. In *International Conference on Machine Learning*. PMLR.
[^RMW2014]: Rezende, D. J., Mohamed, S., & Wierstra, D. (2014, June). Stochastic backpropagation and approximate inference in deep generative models. In *International Conference on Machine Learning*. PMLR.
[^KW2014]: Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. In *International Conference on Learning Representations*.
[^KTRGB2017]: Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. *Journal of machine learning research*.
