
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

## `ADVI` Example 
In this tutorial, we will work with a basic `normal-log-normal` model.
```math
\begin{aligned}
x &\sim \mathrm{LogNormal}\left(\mu_x, \sigma_x^2\right) \\
y &\sim \mathcal{N}\left(\mu_y, \sigma_y^2\right)
\end{aligned}
```
ADVI with `Bijectors.Exp` bijectors is able to infer this model exactly.

Using the `LogDensityProblems` interface, we the model can be defined as follows:
```@example advi
using LogDensityProblems

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
Let's now instantiate the model
```@example advi
using LinearAlgebra

n_dims = 10
μ_x    = randn()
σ_x    = exp.(randn())
μ_y    = randn(n_dims)
σ_y    = exp.(randn(n_dims))
model  = NormalLogNormal(μ_x, σ_x, μ_y, Diagonal(σ_y.^2));
```

Since the `y` follows a log-normal prior, its support is bounded to be the positive half-space ``\mathbb{R}_+``.
Thus, we will use [Bijectors](https://github.com/TuringLang/Bijectors.jl) to match the support of our target posterior and the variational approximation.
```@example advi
using Bijectors

function Bijectors.bijector(model::NormalLogNormal)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    Bijectors.Stacked(
        Bijectors.bijector.([LogNormal(μ_x, σ_x), MvNormal(μ_y, Σ_y)]),
        [1:1, 2:1+length(μ_y)])
end

b   = Bijectors.bijector(model);
b⁻¹ = inverse(b)
```

Let's now load `AdvancedVI`.
Since ADVI relies on automatic differentiation (AD), hence the "AD" in "ADVI", we need to load an AD library, *before* loading `AdvancedVI`.
Also, the selected AD framework needs to be communicated to `AdvancedVI` using the [ADTypes](https://github.com/SciML/ADTypes.jl) interface.
Here, we will use `ForwardDiff`, which can be selected by later passing `ADTypes.AutoForwardDiff()`.
```@example advi
using Optimisers
using ADTypes, ForwardDiff
import AdvancedVI as AVI
```
We now need to select 1. a variational objective, and 2. a variational family.
Here, we will use the [`ADVI` objective](@ref advi), which expects an object implementing the [`LogDensityProblems`](https://github.com/tpapp/LogDensityProblems.jl) interface, and the inverse bijector.
```@example advi
n_montecaro = 10;
objective   = AVI.ADVI(model, n_montecaro; invbij = b⁻¹)
```
For the variational family, we will use the classic mean-field Gaussian family.
```@example advi
d = LogDensityProblems.dimension(model);
μ = randn(d);
L = Diagonal(ones(d));
q = AVI.VIMeanFieldGaussian(μ, L)
```
Passing `objective` and the initial variational approximation `q` to `optimize` performs inference.
```@example advi
n_max_iter  = 10^4
q, stats, _ = AVI.optimize(
    objective,
    q,
    n_max_iter;
    adbackend = AutoForwardDiff(),
    optimizer = Optimisers.Adam(1e-3)
); 
```

The selected inference procedure stores per-iteration statistics into `stats`.
For instance, the ELBO can be ploted as follows:
```@example advi
using Plots

t = [stat.iteration for stat ∈ stats]
y = [stat.elbo for stat ∈ stats]
plot(t, y, label="ADVI", xlabel="Iteration", ylabel="ELBO")
savefig("advi_example_elbo.svg")
nothing
```
![](advi_example_elbo.svg)

Further information can be gathered by defining your own `callback!`.

The final ELBO can be estimated by calling the objective directly with a different number of Monte Carlo samples as follows:
```@example advi
ELBO = objective(q; n_samples=10^4)
```
