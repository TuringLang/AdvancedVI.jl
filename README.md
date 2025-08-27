[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://turinglang.org/AdvancedVI.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.org/AdvancedVI.jl/dev/)
[![ForwardDiff](https://github.com/TuringLang/AdvancedVI.jl/actions/workflows/ForwardDiff.yml/badge.svg?branch=main)](https://github.com/TuringLang/AdvancedVI.jl/actions/workflows/ForwardDiff.yml?query=branch%3Amain)
[![ReverseDiff](https://github.com/TuringLang/AdvancedVI.jl/actions/workflows/ReverseDiff.yml/badge.svg?branch=main)](https://github.com/TuringLang/AdvancedVI.jl/actions/workflows/ReverseDiff.yml?query=branch%3Amain)
[![Mooncake](https://github.com/TuringLang/AdvancedVI.jl/actions/workflows/Mooncake.yml/badge.svg?branch=main)](https://github.com/TuringLang/AdvancedVI.jl/actions/workflows/Mooncake.yml?query=branch%3Amain)
[![Enzyme](https://github.com/TuringLang/AdvancedVI.jl/actions/workflows/Enzyme.yml/badge.svg?branch=main)](https://github.com/TuringLang/AdvancedVI.jl/actions/workflows/Enzyme.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/TuringLang/AdvancedVI.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/TuringLang/AdvancedVI.jl)

# AdvancedVI.jl

[AdvancedVI](https://github.com/TuringLang/AdvancedVI.jl) provides implementations of variational inference (VI) algorithms, which is a family of algorithms aiming for scalable approximate Bayesian inference by leveraging optimization.
`AdvancedVI` is part of the [Turing](https://turinglang.org/stable/) probabilistic programming ecosystem.
The purpose of this package is to provide a common accessible interface for various VI algorithms and utilities so that other packages, e.g. `Turing`, only need to write a light wrapper for integration.
For example, integrating `Turing` with `AdvancedVI.ADVI` only involves converting a `Turing.Model` into a [`LogDensityProblem`](https://github.com/tpapp/LogDensityProblems.jl) and extracting a corresponding `Bijectors.bijector`.

## Basic Example

We will describe a simple example to demonstrate the basic usage of `AdvancedVI`.
`AdvancedVI` works with differentiable models specified through the [`LogDensityProblem`](https://github.com/tpapp/LogDensityProblems.jl) interface.
Let's look at a basic logistic regression example with a hierarchical prior.
For a dataset $(X, y)$ with the design matrix $X \in \mathbb{R}^{n \times d}$ and the response variables $y \in \{0, 1\}^n$, we assume the following data generating process:

$$
\begin{aligned}
\sigma &\sim \text{LogNormal}(0, 1) \\
\beta &\sim \text{Normal}\left(0_d, \sigma^2 \mathrm{I}_d\right) \\
y &\sim \mathrm{BernoulliLogit}\left(X \beta\right)
\end{aligned}
$$

The `LogDensityProblem` corresponding to this model can be constructed as

```julia
using LogDensityProblems: LogDensityProblems
using Distributions
using FillArrays

struct LogReg{XType,YType}
    X::XType
    y::YType
end

function LogDensityProblems.logdensity(model::LogReg, θ)
    (; X, y) = model
    d = size(X, 2)
    β, σ = θ[1:size(X, 2)], θ[end]

    logprior_β = logpdf(MvNormal(Zeros(d), σ), β)
    logprior_σ = logpdf(LogNormal(0, 3), σ)

    logit = X*β
    loglike_y = mapreduce((li, yi) -> logpdf(BernoulliLogit(li), yi), +, logit, y)
    return loglike_y + logprior_β + logprior_σ
end

function LogDensityProblems.dimension(model::LogReg)
    return size(model.X, 2) + 1
end

function LogDensityProblems.capabilities(::Type{<:LogReg})
    return LogDensityProblems.LogDensityOrder{0}()
end;
```

Since the support of `σ` is constrained to be positive and most VI algorithms assume an unconstrained Euclidean support, we need to use a *bijector* to transform `θ`.
We will use [`Bijectors`](https://github.com/TuringLang/Bijectors.jl) for this purpose.
This corresponds to the automatic differentiation variational inference (ADVI) formulation[^KTRGB2017].

```julia
using Bijectors: Bijectors

function Bijectors.bijector(model::LogReg)
    d = size(model.X, 2)
    return Bijectors.Stacked(
        Bijectors.bijector.([MvNormal(Zeros(d), 1.0), LogNormal(0, 3)]),
        [1:d, (d + 1):(d + 1)],
    )
end;
```

A simpler approach would be to use [`Turing`](https://github.com/TuringLang/Turing.jl), where a `Turing.Model` can be automatically be converted into a `LogDensityProblem` and a corresponding `bijector` is automatically generated.

For the dataset, we will use the popular [sonar classification dataset](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks) from the UCI repository.
This can be automatically downloaded using [`OpenML`](https://github.com/JuliaAI/OpenML.jl).
The sonar dataset corresponds to the dataset id 40.

```julia
using OpenML: OpenML
using DataFrames: DataFrames
data = Array(DataFrames.DataFrame(OpenML.load(40)))
X = Matrix{Float64}(data[:, 1:(end - 1)])
y = Vector{Bool}(data[:, end] .== "Mine");
```

Let's apply some basic pre-processing and add an intercept column:

```julia
X = (X .- mean(X; dims=2)) ./ std(X; dims=2)
X = hcat(X, ones(size(X, 1)));
```

The model can now be instantiated as follows:

```julia
model = LogReg(X, y);
```

For the VI algorithm, we will use `KLMinRepGradDescent`:

```julia
using ADTypes, ReverseDiff
using AdvancedVI

alg = KLMinRepGradDescent(ADTypes.AutoReverseDiff());
```

This algorithm minimizes the exclusive/reverse KL divergence via stochastic gradient descent in the (Euclidean) space of the parameters of the variational approximation with the reparametrization gradient[^TL2014][^RMW2014][^KW2014].
This is also commonly referred as automatic differentiation VI, black-box VI, stochastic gradient VI, and so on.

`KLMinRepGradDescent`, in particular, assumes that the target `LogDensityProblem` is differentiable.
If the `LogDensityProblem` has a differentiation [capability](https://www.tamaspapp.eu/LogDensityProblems.jl/dev/#LogDensityProblems.capabilities) of at least first-order, we can take advantage of this.
For this example, we will use `LogDensityProblemsAD` to equip our problem with a first-order capability:

```julia
using DifferentiationInterface: DifferentiationInterface
using LogDensityProblemsAD: LogDensityProblemsAD

model_ad = LogDensityProblemsAD.ADgradient(ADTypes.AutoReverseDiff(), model);
```

For the variational family, we will consider a `FullRankGaussian` approximation:

```julia
using LinearAlgebra

d = LogDensityProblems.dimension(model_ad)
q = FullRankGaussian(zeros(d), LowerTriangular(Matrix{Float64}(I, d, d)))
q = MeanFieldGaussian(zeros(d), Diagonal(ones(d)));
```

The bijector can now be applied to `q` to match the support of the target problem.

```julia
b = Bijectors.bijector(model)
binv = Bijectors.inverse(b)
q_transformed = Bijectors.TransformedDistribution(q, binv);
```

We can now run VI:

```julia
max_iter = 10^3
q, info, _ = AdvancedVI.optimize(alg, max_iter, model_ad, q_transformed;);
```

For more examples and details, please refer to the documentation.

[^TL2014]: Titsias, M., & Lázaro-Gredilla, M. (2014, June). Doubly stochastic variational Bayes for non-conjugate inference. In *International Conference on Machine Learning*. PMLR.
[^RMW2014]: Rezende, D. J., Mohamed, S., & Wierstra, D. (2014, June). Stochastic backpropagation and approximate inference in deep generative models. In *International Conference on Machine Learning*. PMLR.
[^KW2014]: Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. In *International Conference on Learning Representations*.
[^KTRGB2017]: Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. *Journal of machine learning research*.
