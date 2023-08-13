
# [Getting Started with `AdvancedVI`](@id getting_started)

## General Usage
Each VI algorithm should provide the following:
1. A variational family
2. A variational objective

Feeding these two into `optimize` runs the inference procedure.

```@docs
optimize
```

## `ADVI` Example Using `Turing`

```julia
using Turing
using Bijectors
using Optimisers
using ForwardDiff
using ADTypes

import AdvancedVI as AVI

μ_y, σ_y = 1.0, 1.0
μ_z, Σ_z = [1.0, 2.0], [1.0 0.; 0. 2.0]

Turing.@model function normallognormal()
    y ~ LogNormal(μ_y, σ_y)
    z ~ MvNormal(μ_z, Σ_z)
end
model = normallognormal()
b     = Bijectors.bijector(model)
b⁻¹   = inverse(b)
prob  = DynamicPPL.LogDensityFunction(model)
d     = LogDensityProblems.dimension(prob)

μ = randn(d)
L = Diagonal(ones(d))
q = AVI.MeanFieldGaussian(μ, L)

n_max_iter = 10^4
q, stats = AVI.optimize(
    AVI.ADVI(prob, b⁻¹, 10),
    q,
    n_max_iter;
    adbackend = AutoForwardDiff(),
    optimizer = Optimisers.Adam(1e-3)
)
```
