
# [Getting Started with `AdvancedVI`](@id getting_started)

## General Usage
Each VI algorithm provides the followings:
1. Variational families supported by each VI algorithm.
2. A variational objective corresponding to the VI algorithm.
Note that each variational family is subject to its own constraints.
Thus, please refer to the documentation of the variational inference algorithm of interest. 

To use `AdvancedVI`, a user needs to select a `variational family`, `variational objective`,  and feed them into `optimize`.

```@docs
optimize
```

## `ADVI` Example Using `Turing`

In this tutorial, we'll use `Turing` to define a basic `normal-log-normal` model.
ADVI with log bijectors is able to infer this model exactly.
```julia
using Turing

μ_y, σ_y = 1.0, 1.0
μ_z, Σ_z = [1.0, 2.0], [1.0 0.; 0. 2.0]

Turing.@model function normallognormal()
    y ~ LogNormal(μ_y, σ_y)
    z ~ MvNormal(μ_z, Σ_z)
end
model = normallognormal()
```

Since the `y` follows a log-normal prior, its support is bounded to be the positive half-space ``\mathbb{R}_+``.
Thus, we will use [Bijectors](https://github.com/TuringLang/Bijectors.jl) to match the support of our target posterior and the variational approximation.
```julia
using Bijectors

b     = Bijectors.bijector(model)
b⁻¹   = inverse(b)
```

Let's now load `AdvancedVI`.
Since ADVI relies on automatic differentiation (AD), hence the "AD" in "ADVI", we need to load an AD library, *before* loading `AdvancedVI`.
Also, the selected AD framework needs to be communicated to `AdvancedVI` using the [ADTypes](https://github.com/SciML/ADTypes.jl) interface.
Here, we will use `ForwardDiff`, which can be selected by later passing `ADTypes.AutoForwardDiff()`.
```julia
using Optimisers
using ForwardDiff
import AdvancedVI as AVI
```
We now need to select 1. a variational objective, and 2. a variational family.
Here, we will use the [ADVI objective](@ref advi), which expects an object implementing the [`LogDensityProblems`](https://github.com/tpapp/LogDensityProblems.jl) interface, and the inverse bijector.
```julia
prob      = DynamicPPL.LogDensityFunction(model)
objective = AVI.ADVI(prob, b⁻¹, 10),
```
For the variational family, we will use the classic mean-field Gaussian family.
```julia
d = LogDensityProblems.dimension(prob)
μ = randn(d)
L = Diagonal(ones(d))
q = AVI.VIMeanFieldGaussian(μ, L)
```
It now remains to run inverence!
```
n_max_iter = 10^4
q, stats   = AVI.optimize(
    q,
    n_max_iter;
    adbackend = AutoForwardDiff(),
    optimizer = Optimisers.Adam(1e-3)
)
```
