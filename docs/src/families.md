
# [Variational Families](@id families)

## Location-Scale Variational Family

### Description
The [location-scale](https://en.wikipedia.org/wiki/Location%E2%80%93scale_family) variational family is a family of probability distributions, where their sampling process can be represented as
```math
z = C u + m,
```
where ``C`` is the *scale* and ``m`` is the location variational parameter.
This family encompases many 


### Constructors

```@docs
VILocationScale
```

```@docs
VIFullRankGaussian
VIMeanFieldGaussian
```

### Examples

A full-rank variational family can be formed by choosing
```@repl locscale
using AdvancedVI, LinearAlgebra
μ = zeros(2);
L = diagm(ones(2)) |> LowerTriangular;
```

A mean-field variational family can be formed by choosing 
```@repl locscale
μ = zeros(2);
L = ones(2) |> Diagonal;
```

Gaussian variational family:
```@repl locscale
q = VIFullRankGaussian(μ, L)
q = VIMeanFieldGaussian(μ, L)
```

Sudent-T Variational Family:

```@repl locscale
ν = 3
q = VILocationScale(μ, L, StudentT(ν))
```

Multivariate Laplace family:
```@repl locscale
q = VILocationScale(μ, L, Laplace())
```

