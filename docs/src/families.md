
# Variational Families

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

```@repl locscale
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);
```

Gaussian variational family:
```@repl locscale
L = diagm(ones(2)) |> LowerTriangular;
q = VIFullRankGaussian(μ, L)

L = ones(2) |> Diagonal;
q = VIMeanFieldGaussian(μ, L)
```

Sudent-T Variational Family:

```@repl locscale
ν = 3;

# Full-Rank 
L = diagm(ones(2)) |> LowerTriangular;
q = VILocationScale(μ, L, TDist(ν))

# Mean-Field
L = ones(2) |> Diagonal;
q = VILocationScale(μ, L, TDist(ν))
```

Multivariate Laplace family:
```@repl locscale
# Full-Rank 
L = diagm(ones(2)) |> LowerTriangular;
q = VILocationScale(μ, L, Laplace())

# Mean-Field
L = ones(2) |> Diagonal;
q = VILocationScale(μ, L, Laplace())
```

