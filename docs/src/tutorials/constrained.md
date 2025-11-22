# [Dealing with Constrained Posteriors](@id constrained)

In this tutorial, we will demonstrate how to deal with constrained posteriors in more detail.
Formally, by constrained posteriors, we mean that the target posterior has a density defined over a space that does not span the "full" Euclidean space $\mathbb{R}^d$:

```math
\pi : \mathcal{X} \to \mathbb{R}_{> 0} ,
```

where $\mathcal{X} \subset \mathbb{R}^d$ but not $\mathcal{X} = \mathbb{R}^d$.

For instance, consider the basic hierarchical model for estimating the mean of the data $y_1, \ldots, y_n$:

```math
\begin{aligned}
    \sigma &\sim \operatorname{LogNormal}(\alpha, \beta) \\
    \mu &\sim \operatorname{Normal}(0, \sigma) \\
    y_i &\sim \operatorname{Normal}(\mu, \sigma) .
\end{aligned}
```

The corresponding posterior

```math
\pi(\mu, \sigma \mid y_1, \ldots, y_n)
=
\operatorname{LogNormal}(\sigma; \alpha, \beta)
\operatorname{Normal}(\mu; 0, \sigma)
\prod_{i=1}^n \operatorname{Normal}(y_i; \mu, \sigma)
```

has a density with respect to the space

```math
    \mathcal{X} = \mathbb{R}_{> 0} \times \mathbb{R} .
```

There are also more complicated examples of constrained spaces.
For example, a $k$-dimensional variable with a Dirichlet prior will be constrained to live on a $k$-dimensional simplex.

Now, most algorithms provided by `AdvancedVI`, such as:

  - `KLMinRepGradDescent`
  - `KLMinRepGradProxDescent`
  - `KLMinNaturalGradDescent`
  - `FisherMinBatchMatch`

tend to assume the target posterior is defined over the whole Euclidean space $\mathbb{R}^d$.
Therefore, to apply these algorithms, we need to do something about the constraints.
We will describe some recommended ways of doing this.

## Transforming the Posterior

The most widely applicable way is to transform the posterior $\pi : \mathcal{X} \to \mathbb{R}_{>0}$ to be unconstrained.
That is, consider some bijective map $b : \mathcal{X} \to \mathbb{R}^{d}$ between the $\mathcal{X}$ and the associated Euclidean space $\mathbb{R}^{d}$.
Using the inverse of the map $b^{-1}$ and its Jacobian $\mathrm{J}_{b^{-1}}$, we can apply a change of variable to the posterior and obtain its unconstrained counterpart

```math
\pi_{b^{-1}}(\eta) : \mathbb{R}^d \to \mathbb{R}_{>0} = \pi(b^{-1}(\eta)) {\lvert  \mathrm{J}_{b^{-1}}(\eta)  \rvert} .
```

This idea popularized by Stan[^CGHetal2017] and Tensorflow probability[^DLTetal2017] is, in fact, how most probabilistic programming frameworks enable the use of off-the-shelf Markov chain Monte Carlo algorithms.
In the context of variational inference, we will first approximate the unconstrained posterior as

```math
q^* = \arg\min_{q \in \mathcal{Q}} \;\; \mathrm{D}(q, \pi_{b^{-1}}) .
```

and then transform the optimal unconstrained approximation $q^*$ to be constrained by again applying a change of variable as

```math
q_{b}^* : \mathcal{X} \to \mathbb{R}_{>0} = q(b(z)) {\lvert \mathrm{J}_{b}(z) \rvert} .
```

Sampling from $q_{b}^*$ amounts to pushing each sample from $q$ into $b^{-1}$:

```math
z \sim q_{b}^* \quad\Leftrightarrow\quad z \stackrel{\mathrm{d}}{=} b^{-1}(\eta) ; \quad \eta \sim q^* .
```

The idea of applying a change-of-variable to the variational approximation to match a constrained posterior was popularized by the automatic differentiation VI[^KTRGB2017].

[^KTRGB2017]: Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. Journal of machine learning research, 18(14), 1-45.
Now, there are two ways how to do this in Julia.
First, let's define the constrained posterior example above using the `LogDensityProblems` interface for illustration:

```@example constraints
using LogDensityProblems

struct Mean
    y::Vector{Float64}
end

function LogDensityProblems.logdensity(prob::Mean, θ)
    μ, σ = θ[1], θ[2]
    ℓp_μ = logpdf(Normal(0, σ), μ)
    ℓp_σ = logpdf(LogNormal(0, 3), σ)
    ℓl_y = mapreduce(yi -> logpdf(Normal(μ, σ), yi), +, prob.y)
    return ℓp_μ + ℓp_σ + ℓl_y
end

LogDensityProblems.dimension(::Mean) = 2

LogDensityProblems.capabilities(::Type{Mean}) = LogDensityProblems.LogDensityOrder{0}()

n_data = 30
prob = Mean(randn(n_data))
nothing
```

We need to find the right transformation associated with a `LogNormal` prior.
Most of the common bijective transformations can be found in [`Bijectors.jl`](https://github.com/TuringLang/Bijectors.jl) package[^FXTYG2020].
See the following:

```@example constraints
using Bijectors

b_σ = Bijectors.bijector(LogNormal(0, 1))
```

and the inverse transformation can be obtained as

```@example constraints
binv_σ = Bijectors.inverse(b_σ)
```

Multiple bijectors can also be stacked to form a joint bijector using `Bijectors.Stacked`.
For example:

```@example constraints
function Bijectors.bijector(::Mean)
    return Bijectors.Stacked(
        Bijectors.bijector.([Normal(0, 1), LogNormal(1, 1)]), [1:1, 2:2]
    )
end

b = Bijectors.bijector(prob)
binv = Bijectors.inverse(b)
```

Refer to the documentation of `Bijectors.jl` for more details.

## Wrap the `LogDensityProblem`

The most general and easy way to obtain an unconstrained posterior using a `Bijector` is to wrap our original `LogDensityProblem` to form a new `LogDensityProblem`.
This approach only requires the user to implement the model-specific `Bijectors.bijector` function as above.
The rest can be done by simply copy-pasting the code below:

```@example constraints
struct TransformedLogDensityProblem{Prob,BInv}
    prob::Prob
    binv::BInv
end

function TransformedLogDensityProblem(prob)
    b = Bijectors.bijector(prob)
    binv = Bijectors.inverse(b)
    return TransformedLogDensityProblem{typeof(prob),typeof(binv)}(prob, binv)
end

function LogDensityProblems.logdensity(prob_trans::TransformedLogDensityProblem, θ_trans)
    (; prob, binv) = prob_trans
    θ, logabsdetjac = Bijectors.with_logabsdet_jacobian(binv, θ_trans)
    return LogDensityProblems.logdensity(prob, θ) + logabsdetjac
end

function LogDensityProblems.dimension(prob_trans::TransformedLogDensityProblem)
    (; prob, binv) = prob_trans
    b = Bijectors.inverse(binv)
    d = LogDensityProblems.dimension(prob)
    return prod(Bijectors.output_size(b, (d,)))
end

function LogDensityProblems.capabilities(
    ::Type{TransformedLogDensityProblem{Prob,BInv}}
) where {Prob,BInv}
    return LogDensityProblems.capabilities(Prob)
end
nothing
```

Wrapping `prob` with `TransformedLogDensityProblem` yields our unconstrained posterior.

```@example constraints
prob_trans = TransformedLogDensityProblem(prob)

x = randn(LogDensityProblems.dimension(prob_trans)) # sample on an unconstrained support
LogDensityProblems.logdensity(prob_trans, x)
```

We can also wrap `prob_trans` with `LogDensityProblemsAD.ADGradient` to make it differentiable.

```@example constraints
using LogDensityProblemsAD
using ADTypes, ReverseDiff

prob_trans_ad = LogDensityProblemsAD.ADgradient(
    ADTypes.AutoReverseDiff(; compile=true), prob_trans; x=randn(2)
)
```

Let's now run VI to verify that it works.
Here, we will use `FisherMinBatchMatch`, which expects an unconstrained posterior.

```@example constraints
using AdvancedVI
using LinearAlgebra

d = LogDensityProblems.dimension(prob_trans_ad)
q = FullRankGaussian(zeros(d), LowerTriangular(Matrix{Float64}(0.6*I, d, d)))

q_opt, info, _ = AdvancedVI.optimize(
    FisherMinBatchMatch(), 100, prob_trans_ad, q; show_progress=false
)
nothing
```

We have now obtained a variational approximation `q_opt` of the unconstrained posterior associated with `prob_trans`.
It remains to transform `q_opt` back to the constrained space we were originally interested in.
This can be done by wrapping it into a `Bijectors.TransformedDistribution`.

```@example constraints
q_opt_trans = Bijectors.TransformedDistribution(q_opt, binv)
```

```@example constraints
using Plots

x = rand(q_opt_trans, 1000)

Plots.stephist(x[2, :]; normed=true, xlabel="Posterior of σ", label=nothing, xlims=(0, 2))
Plots.vline!([1.0]; label="True Value")
savefig("constrained_histogram.svg")
```

![](constrained_histogram.svg)

We can see that the transformed posterior is indeed a meaningful approximation of the original posterior $\pi(\sigma \mid y_1, \ldots, y_n)$ we were interested in.

## Bake a Bijector into the `LogDensityProblem`

A problem with the general approach above is that automatically differentiating through `TransformedLogDensityProblem` can be a bit inefficient (due to `Stacked`), especially with reverse-mode AD.
Therefore, another effective but less automatic approach is to bake the transformation and Jacobian adjustment into the `LogDensityProblem` itself.
Here is an example for our mean estimation model:

```@example constraints
struct MeanTransformed{BInvS}
    y::Vector{Float64}
    binv_σ::BInvS
end

function MeanTransformed(y::Vector{Float64})
    binv_σ = Bijectors.inverse(Bijectors.bijector(LogNormal(0, 3)))
    return MeanTransformed(y, binv_σ)
end

function LogDensityProblems.logdensity(prob::MeanTransformed, θ)
    (; y, binv_σ) = prob
    μ = θ[1]

    # Apply bijector and compute Jacobian
    σ, ℓabsdetjac_σ = with_logabsdet_jacobian(binv_σ, θ[2])

    ℓp_μ = logpdf(Normal(0, σ), μ)
    ℓp_σ = logpdf(LogNormal(0, 3), σ)
    ℓl_y = mapreduce(yi -> logpdf(Normal(μ, σ), yi), +, prob.y)
    return ℓp_μ + ℓp_σ + ℓl_y + ℓabsdetjac_σ
end

LogDensityProblems.dimension(::MeanTransformed) = 2

function LogDensityProblems.capabilities(::Type{MeanTransformed})
    LogDensityProblems.LogDensityOrder{0}()
end

n_data = 30
prob_bakedtrans = MeanTransformed(randn(n_data))
nothing
```

Now, `prob_bakedtrans` can be used identically as `prob_trans` above.
For problems with larger dimensions, however, baking the bijector into the problem as above could be significantly more efficient.

[^CGHetal2017]: Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M., ... & Riddell, A. (2017). Stan: A probabilistic programming language. Journal of statistical software, 76, 1-32.
[^DLTetal2017]: Dillon, J. V., Langmore, I., Tran, D., Brevdo, E., Vasudevan, S., Moore, D., ... & Saurous, R. A. (2017). Tensorflow distributions. arXiv preprint arXiv:1711.10604.
[^FXTYG2020]: Fjelde, T. E., Xu, K., Tarek, M., Yalburgi, S., & Ge, H. (2020, February). Bijectors. jl: Flexible transformations for probability distributions. In Symposium on Advances in Approximate Bayesian Inference (pp. 1-17). PMLR.
