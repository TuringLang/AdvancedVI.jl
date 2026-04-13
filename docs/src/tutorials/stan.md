# Stan Models

Since `AdvancedVI` supports the [`LogDensityProblem`](https://github.com/tpapp/LogDensityProblems.jl) interface, it can also be used with Stan models through [`StanLogDensityProblems`](https://github.com/sethaxen/StanLogDensityProblems.jl) interface.
Specifically, `StanLogDensityProblems` wraps any Stan model into a `LogDensityProblem` using [`BridgeStan`](https://github.com/roualdes/bridgestan).

## Problem Setup

Recall the hierarchical logistic regression example in the [Basic Example](@ref basic).
Here, we will define the same model in Stan.

```@example stan
model_src = """
data {
  int<lower=0> N;
  int<lower=0> D;
  matrix[N,D] X;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  vector[D] beta;
  real<lower=1e-4> sigma;
}
model {
  sigma ~ lognormal(0, 1);
  beta ~ normal(0, sigma);
  y ~ bernoulli_logit(X * beta);
}
"""
nothing
```

We also need to prepare the data. To keep the tutorial self-contained, we reuse a synthetic binary classification dataset.

```@example stan
using Random
using Statistics

Random.seed!(1)
n_data = 208
n_features = 60
X = randn(n_data, n_features)
β_true = 0.2 .* collect(range(-1, 1; length=n_features))
logits = X * β_true
y = Vector{Int}(rand(n_data) .< @. inv(1 + exp(-logits)))
X = (X .- mean(X; dims=1)) ./ std(X; dims=1)
X = hcat(X, ones(size(X, 1)))

stan_data = (X=transpose(X), y=y, N=size(X, 1), D=size(X, 2))
nothing
```

Since `StanLogDensityProblems` expects files for both the model and the data, we need to store both on the file system.

```@example stan
using JSON: JSON

open("logistic_model.stan", "w") do io
    println(io, model_src)
end
open("logistic_data.json", "w") do io
    println(io, JSON.json(stan_data))
end
nothing
```

## Inference via AdvancedVI

We can now call `StanLogDensityProblems` to recieve a `LogDensityProblem`.

```@example stan
using StanLogDensityProblems: StanLogDensityProblems

model = StanLogDensityProblems.StanProblem("logistic_model.stan", "logistic_data.json")
nothing
```

The rest is the same as all `LogDensityProblem` with the exception of how to deal with constrainted variables: Since `StanLogDensityProblems` automatically transforms the support of the target problem to be unconstrained, we do not need to involve `Bijectors`.

```@example stan
using ADTypes
using AdvancedVI
using LinearAlgebra
using LogDensityProblems
using Plots

alg = KLMinRepGradDescent(ADTypes.AutoMooncake(); operator=ClipScale())

d = LogDensityProblems.dimension(model)
q = FullRankGaussian(zeros(d), LowerTriangular(Matrix{Float64}(0.37 * I, d, d)))

max_iter = 2_000
q_out, info, _ = AdvancedVI.optimize(alg, max_iter, model, q; show_progress=false)

plot(
    [i.iteration for i in info],
    [i.elbo for i in info];
    xlabel="Iteration",
    ylabel="ELBO",
    label=nothing,
    size=(700, 420),
)
```

From variational posterior `q_out` we can draw samples from the unconstrained support of the model.
To convert the samples back to the original (constrained) support of the model, it suffices to call [BridgeStan.param_constrain](https://roualdes.us/bridgestan/latest/languages/julia.html#BridgeStan.param_constrain).
