# [`KLMinRepGradDescent`](@id klminrepgraddescent)

## Description

This algorithm aims to minimize the exclusive (or reverse) Kullback-Leibler (KL) divergence via stochastic gradient descent in the space of parameters.
Specifically, it uses the the *reparameterization gradient estimator*.
As a result, this algorithm is best applicable when the target log-density is differentiable and the sampling process of the variational family is differentiable.
(See the [methodology section](@ref klminrepgraddescent_method) for more details.)
This algorithm is also commonly referred to as automatic differentiation variational inference, black-box variational inference with the reparameterization gradient, and stochastic gradient variational inference.
`KLMinRepGradDescent` is also an alias of `ADVI` .

```@docs
KLMinRepGradDescent
```

## [Methodology](@id klminrepgraddescent_method)

This algorithm aims to solve the problem

```math
  \mathrm{minimize}_{q \in \mathcal{Q}}\quad \mathrm{KL}\left(q, \pi\right)
```

where $\mathcal{Q}$ is some family of distributions, often called the variational family, by running stochastic gradient descent in the (Euclidean) space of parameters.
That is, for all $$q_{\lambda} \in \mathcal{Q}$$, we assume $$q_{\lambda}$$ there is a corresponding vector of parameters $$\lambda \in \Lambda$$, where the space of parameters is Euclidean such that $$\Lambda \subset \mathbb{R}^p$$.

Since we usually only have access to the unnormalized densities of the target distribution $\pi$, we don't have direct access to the KL divergence.
Instead, the ELBO maximization strategy maximizes a surrogate objective, the *evidence lower bound* (ELBO; [^JGJS1999])

```math
  \mathrm{ELBO}\left(q\right) \triangleq \mathbb{E}_{\theta \sim q} \log \pi\left(\theta\right) + \mathbb{H}\left(q\right),
```

which is equivalent to the KL up to an additive constant (the evidence).

Algorithmically, `KLMinRepGradDescent` iterates the step

```math
  \lambda_{t+1} = \mathrm{operator}\big(
      \lambda_{t} + \gamma_t \widehat{\nabla_{\lambda} \mathrm{ELBO}} (q_{\lambda_t}) 
  \big) , 
```

where $\widehat{\nabla \mathrm{ELBO}}(q_{\lambda})$ is the reparameterization gradient estimate[^HC1983][^G1991][^R1992][^P1996] of the ELBO gradient and $$\mathrm{operator}$$ is an optional operator (*e.g.* projections, identity mapping).

The reparameterization gradient, also known as the push-in gradient or the pathwise gradient, was introduced to VI in [^TL2014][^RMW2014][^KW2014].
For the variational family $$\mathcal{Q}$$, suppose the process of sampling from $$q_{\lambda} \in \mathcal{Q}$$ can be described by some differentiable reparameterization function $$T_{\lambda}$$ and a *base distribution* $$\varphi$$ independent of $$\lambda$$ such that

```math
z \sim  q_{\lambda} \qquad\Leftrightarrow\qquad
z \stackrel{d}{=} T_{\lambda}\left(\epsilon\right);\quad \epsilon \sim \varphi \; .
```

In these cases, denoting the target log denstiy as $\log \pi$, we can effectively estimate the gradient of the ELBO by directly differentiating the stochastic estimate of the ELBO objective

```math
  \widehat{\mathrm{ELBO}}\left(q_{\lambda}\right) = \frac{1}{M}\sum^M_{m=1} \log \pi\left(T_{\lambda}\left(\epsilon_m\right)\right) + \mathbb{H}\left(q_{\lambda}\right),
```

where $$\epsilon_m \sim \varphi$$ are Monte Carlo samples.

[^JGJS1999]: Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine learning, 37, 183-233.
[^HC1983]: Ho, Y. C., & Cao, X. (1983). Perturbation analysis and optimization of queueing networks. Journal of optimization theory and Applications, 40(4), 559-582.
[^G1991]: Glasserman, P. (1991). Gradient estimation via perturbation analysis (Vol. 116). Springer Science & Business Media.
[^R1992]: Rubinstein, R. Y. (1992). Sensitivity analysis of discrete event systems by the “push out” method. Annals of Operations Research, 39(1), 229-250.
[^P1996]: Pflug, G. C. (1996). Optimization of stochastic models: the interface between simulation and optimization (Vol. 373). Springer Science & Business Media.
[^TL2014]: Titsias, M., & Lázaro-Gredilla, M. (2014). Doubly stochastic variational Bayes for non-conjugate inference. In *International Conference on Machine Learning*.
[^RMW2014]: Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. In *International Conference on Machine Learning*.
[^KW2014]: Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. In *International Conference on Learning Representations*.
## [Handling Constraints with `Bijectors`](@id bijectors)

As mentioned in the docstring, `KLMinRepGradDescent` assumes that the variational approximation $$q_{\lambda}$$ and the target distribution $$\pi$$ have the same support for all $$\lambda \in \Lambda$$.
However, in general, it is most convenient to use variational families that have the whole Euclidean space $$\mathbb{R}^d$$ as their support.
This is the case for the [location-scale distributions](@ref locscale) provided by `AdvancedVI`.
For target distributions which the support is not the full $$\mathbb{R}^d$$, we can apply some transformation $$b$$ to $$q_{\lambda}$$ to match its support such that

```math
z \sim  q_{b,\lambda} \qquad\Leftrightarrow\qquad
z \stackrel{d}{=} b^{-1}\left(\eta\right);\quad \eta \sim q_{\lambda},
```

where $$b$$ is often called a *bijector*, since it is often chosen among bijective transformations.
This idea is known as automatic differentiation VI[^KTRGB2017] and has subsequently been improved by Tensorflow Probability[^DLTBV2017].
In Julia, [Bijectors.jl](https://github.com/TuringLang/Bijectors.jl)[^FXTYG2020] provides a comprehensive collection of bijections.

[^KTRGB2017]: Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. *Journal of Machine Learning Research*, 18(14), 1-45.
[^DLTBV2017]: Dillon, J. V., Langmore, I., Tran, D., Brevdo, E., Vasudevan, S., Moore, D., ... & Saurous, R. A. (2017). Tensorflow distributions. arXiv.
[^FXTYG2020]: Fjelde, T. E., Xu, K., Tarek, M., Yalburgi, S., & Ge, H. (2020,. Bijectors. jl: Flexible transformations for probability distributions. In *Symposium on Advances in Approximate Bayesian Inference*.
    One caveat of ADVI is that, after applying the bijection, a Jacobian adjustment needs to be applied.
    That is, the objective is now
```math
\mathrm{ADVI}\left(\lambda\right)
\triangleq
\mathbb{E}_{\eta \sim q_{\lambda}}\left[
  \log \pi\left( b^{-1}\left( \eta \right) \right)
  + \log \lvert J_{b^{-1}}\left(\eta\right) \rvert
\right]
+ \mathbb{H}\left(q_{\lambda}\right)
```

This is automatically handled by `AdvancedVI` through `TransformedDistribution` provided by `Bijectors.jl`.
See the following example:

```julia
using Bijectors
q = MeanFieldGaussian(μ, L)
b = Bijectors.bijector(dist)
binv = inverse(b)
q_transformed = Bijectors.TransformedDistribution(q, binv)
```

By passing `q_transformed` to `optimize`, the Jacobian adjustment for the bijector `b` is automatically applied.
(See the [Basic Example](@ref basic) for a fully working example.)

## [Entropy Gradient Estimators](@id entropygrad)

For the gradient of the entropy term, we provide three choices with varying requirements.
The user can select the entropy estimator by passing it as a keyword argument when constructing the algorithm object.

| Estimator                   | `entropy(q)` | `logpdf(q)` | Type                             |
|:--------------------------- |:------------:|:-----------:|:-------------------------------- |
| `ClosedFormEntropy`         | required     |             | Deterministic                    |
| `MonteCarloEntropy`         |              | required    | Monte Carlo                      |
| `StickingTheLandingEntropy` |              | required    | Monte Carlo with control variate |

The requirements mean that either `Distributions.entropy` or `Distributions.logpdf` need to be implemented for the choice of variational family.
In general, the use of `ClosedFormEntropy` is recommended whenever possible.
If `entropy` is not available, then `StickingTheLandingEntropy` is recommended.
See the following section for more details.

### The `StickingTheLandingEntropy` Estimator

The `StickingTheLandingEntropy`, or STL estimator, is a control variate approach [^RWD2017].

```@docs
StickingTheLandingEntropy
```

It occasionally results in lower variance when ``\pi \approx q_{\lambda}``, and higher variance when ``\pi \not\approx q_{\lambda}``.
The conditions for which the STL estimator results in lower variance is still an active subject for research.

The main downside of the STL estimator is that it needs to evaluate and differentiate the log density of ``q_{\lambda}``, `logpdf(q)`, in every iteration.
Depending on the variational family, this might be computationally inefficient or even numerically unstable.
For example, if ``q_{\lambda}`` is a Gaussian with a full-rank covariance, a back-substitution must be performed at every step, making the per-iteration complexity ``\mathcal{O}(d^3)`` and reducing numerical stability.

```@setup repgradelbo
using Bijectors
using FillArrays
using LinearAlgebra
using LogDensityProblems
using Plots
using Random

using Optimisers
using ADTypes, ForwardDiff, ReverseDiff
using AdvancedVI

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

function LogDensityProblems.logdensity_and_gradient(model::NormalLogNormal, θ)
    return (
        LogDensityProblems.logdensity(model, θ),
        ForwardDiff.gradient(Base.Fix1(LogDensityProblems.logdensity, model), θ),
    )
end

function LogDensityProblems.dimension(model::NormalLogNormal)
    length(model.μ_y) + 1
end

function LogDensityProblems.capabilities(::Type{<:NormalLogNormal})
    LogDensityProblems.LogDensityOrder{1}()
end

n_dims = 10
μ_x    = 2.0
σ_x    = 0.3
μ_y    = Fill(2.0, n_dims)
σ_y    = Fill(1.0, n_dims)
model  = NormalLogNormal(μ_x, σ_x, μ_y, Diagonal(σ_y.^2));

d  = LogDensityProblems.dimension(model);
μ  = zeros(d);
L  = Diagonal(ones(d));
q0 = AdvancedVI.MeanFieldGaussian(μ, L)

function Bijectors.bijector(model::NormalLogNormal)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    Bijectors.Stacked(
        Bijectors.bijector.([LogNormal(μ_x, σ_x), MvNormal(μ_y, Σ_y)]),
        [1:1, 2:1+length(μ_y)])
end
```

In this example, the true posterior is contained within the variational family.
This setting is known as "perfect variational family specification."
In this case, `KLMinRepGradDescent` with `StickingTheLandingEntropy` is the only estimator known to converge exponentially fast ("linear convergence") to the true solution.

Recall that the original ADVI objective with a closed-form entropy (CFE) is given as follows:

```@example repgradelbo
n_montecarlo = 16;
b = Bijectors.bijector(model);
binv = inverse(b)

q0_trans = Bijectors.TransformedDistribution(q0, binv)

cfe = KLMinRepGradDescent(
    AutoReverseDiff();
    entropy=ClosedFormEntropy(),
    optimizer=Adam(1e-2),
    operator=ClipScale(),
)
nothing
```

The repgradelbo estimator can instead be created as follows:

```@example repgradelbo
stl = KLMinRepGradDescent(
    AutoReverseDiff();
    entropy=StickingTheLandingEntropy(),
    optimizer=Adam(1e-2),
    operator=ClipScale(),
)
nothing
```

```@setup repgradelbo
max_iter = 3*10^3

function callback(; params, restructure, kwargs...)
    q = restructure(params).dist
    dist2 = sum(abs2, q.location - vcat([μ_x], μ_y)) 
        + sum(abs2, diag(q.scale) - vcat(σ_x, σ_y))
    (dist = sqrt(dist2),)
end

_, info_cfe, _ = AdvancedVI.optimize(
    cfe,
    max_iter,
    model,
    q0_trans;
    show_progress = false,
    callback      = callback,
); 

_, info_stl, _ = AdvancedVI.optimize(
    stl,
    max_iter,
    model,
    q0_trans;
    show_progress = false,
    callback      = callback,
); 

_, info_stl, _ = AdvancedVI.optimize(
    stl,
    max_iter,
    model,
    q0_trans;
    show_progress = false,
    callback      = callback,
); 

t        = [i.iteration for i in info_cfe]
elbo_cfe = [i.elbo      for i in info_cfe]
elbo_stl = [i.elbo      for i in info_stl]
dist_cfe = [i.dist      for i in info_cfe]
dist_stl = [i.dist      for i in info_stl]
plot( t, elbo_cfe, label="BBVI CFE", xlabel="Iteration", ylabel="ELBO")
plot!(t, elbo_stl, label="BBVI STL", xlabel="Iteration", ylabel="ELBO")
savefig("advi_stl_elbo.svg")

plot( t, dist_cfe, label="BBVI CFE", xlabel="Iteration", ylabel="distance to optimum", yscale=:log10)
plot!(t, dist_stl, label="BBVI STL", xlabel="Iteration", ylabel="distance to optimum", yscale=:log10)
savefig("advi_stl_dist.svg")
nothing
```

![](advi_stl_elbo.svg)

We can see that the noise of the repgradelbo estimator becomes smaller as VI converges.
However, the speed of convergence may not always be significantly different.
Also, due to noise, just looking at the ELBO may not be sufficient to judge which algorithm is better.
This can be made apparent if we measure convergence through the distance to the optimum:

![](advi_stl_dist.svg)

We can see that STL kicks-in at later stages of optimization.
Therefore, when STL "works", it yields a higher accuracy solution even on large stepsizes.
However, whether STL works or not highly depends on the problem[^KMG2024].
Furthermore, in a lot of cases, a low-accuracy solution may be sufficient.

[^RWD2017]: Roeder, G., Wu, Y., & Duvenaud, D. K. (2017). Sticking the landing: Simple, lower-variance gradient estimators for variational inference. Advances in Neural Information Processing Systems, 30.
[^KMG2024]: Kim, K., Ma, Y., & Gardner, J. (2024). Linear Convergence of Black-Box Variational Inference: Should We Stick the Landing?. In International Conference on Artificial Intelligence and Statistics (pp. 235-243). PMLR.
## Advanced Usage

There are two major ways to customize the behavior of `KLMinRepGradDescent`

  - Customize the `Distributions` functions: `rand(q)`, `entropy(q)`, `logpdf(q)`.
  - Customize `AdvancedVI.reparam_with_entropy`.

It is generally recommended to customize `rand(q)`, `entropy(q)`, `logpdf(q)`, since it will easily compose with other functionalities provided by `AdvancedVI`.

The most advanced way is to customize `AdvancedVI.reparam_with_entropy`.
In particular, `reparam_with_entropy` is the function that invokes `rand(q)`, `entropy(q)`, `logpdf(q)`.
Thus, it is the most general way to override the behavior of `RepGradELBO`.

```@docs
AdvancedVI.reparam_with_entropy
```

To illustrate how we can customize the `rand(q)` function, we will implement quasi-Monte-Carlo variational inference[^BWM2018].
Consider the case where we use the `MeanFieldGaussian` variational family.
In this case, it suffices to override its `rand` specialization as follows:

```@example repgradelbo
using QuasiMonteCarlo
using StatsFuns

qmcrng = SobolSample(; R=OwenScramble(; base=2, pad=32))

function Distributions.rand(
    rng::AbstractRNG, q::MvLocationScale{<:Diagonal,D,L}, num_samples::Int
) where {L,D}
    (; location, scale, dist) = q
    n_dims = length(location)
    scale_diag = diag(scale)
    unif_samples = QuasiMonteCarlo.sample(num_samples, length(q), qmcrng)
    std_samples = norminvcdf.(unif_samples)
    return scale_diag .* std_samples .+ location
end
nothing
```

(Note that this is a quick-and-dirty example, and there are more sophisticated ways to implement this.)

```@setup repgradelbo
_, info_qmc, _ = AdvancedVI.optimize(
    KLMinRepGradDescent(AutoReverseDiff(); n_samples=n_montecarlo, optimizer=Adam(1e-2), operator=ClipScale()),
    max_iter,
    model,
    q0_trans;
    show_progress = false,
    callback      = callback,
); 

t        = [i.iteration for i in info_qmc]
elbo_qmc = [i.elbo      for i in info_qmc]
dist_qmc = [i.dist      for i in info_qmc]
plot( t, elbo_cfe, label="BBVI CFE",     xlabel="Iteration", ylabel="ELBO")
plot!(t, elbo_qmc, label="BBVI CFE QMC", xlabel="Iteration", ylabel="ELBO")
savefig("advi_qmc_elbo.svg")

plot( t, dist_cfe, label="BBVI CFE",     xlabel="Iteration", ylabel="distance to optimum", yscale=:log10)
plot!(t, dist_qmc, label="BBVI CFE QMC", xlabel="Iteration", ylabel="distance to optimum", yscale=:log10)
savefig("advi_qmc_dist.svg")

# The following definition is necessary to revert the behavior of `rand` so that 
# the example in example.md works with the regular non-QMC estimator.
function Distributions.rand(
    rng::AbstractRNG, q::MvLocationScale{<:Diagonal, D, L}, num_samples::Int
) where {L, D}
    (; location, scale, dist) = q 
    n_dims       = length(location)
    scale_diag   = diag(scale)
    scale_diag.*rand(rng, dist, n_dims, num_samples) .+ location
end
nothing
```

By plotting the ELBO, we can see the effect of quasi-Monte Carlo.
![](advi_qmc_elbo.svg)
We can see that quasi-Monte Carlo results in much lower variance than naive Monte Carlo.
However, similarly to the STL example, just looking at the ELBO is often insufficient to really judge performance.
Instead, let's look at the distance to the global optimum:

![](advi_qmc_dist.svg)

QMC yields an additional order of magnitude in accuracy.
Also, unlike STL, it ever-so slightly accelerates convergence.
This is because quasi-Monte Carlo uniformly reduces variance, unlike STL, which reduces variance only near the optimum.

[^BWM2018]: Buchholz, A., Wenzel, F., & Mandt, S. (2018). Quasi-monte carlo variational inference. In *International Conference on Machine Learning*.
