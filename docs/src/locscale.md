
# [Location-Scale Variational Family](@id locscale)

## Introduction
The [location-scale](https://en.wikipedia.org/wiki/Location%E2%80%93scale_family) variational family is a family of probability distributions, where their sampling process can be represented as
```math
z \sim  q_{\lambda} \qquad\Leftrightarrow\qquad
z \stackrel{d}{=} C u + m;\quad u \sim \varphi
```
where ``C`` is the *scale*, ``m`` is the location, and ``\varphi`` is the *base distribution*.
``m`` and ``C`` form the variational parameters ``\lambda = (m, C)`` of ``q_{\lambda}``. 
The location-scale family encompases many practical variational families, which can be instantiated by setting the *base distribution* of ``u`` and the structure of ``C``.

The probability density is given by
```math
  q_{\lambda}(z) = {|C|}^{-1} \varphi(C^{-1}(z - m))
```
and the entropy is given as
```math
  \mathcal{H}(q_{\lambda}) = \mathcal{H}(\varphi) + \log |C|,
```
where ``\mathcal{H}(\varphi)`` is the entropy of the base distribution.
Notice the ``\mathcal{H}(\varphi)`` does not depend on ``\log |C|``.
The derivative of the entropy with respect to ``\lambda`` is thus independent of the base distribution.

!!! warning
	`LocationScale` and its specializations such as `VIFullRankGaussian` and `VIMeanFieldGaussian` are inefficient with forward-mode differentiation packages like `ForwardDiff`. Especially, they scale poorly with the number of dimensions. Please use reverse-mode differentation packages such as `ReverseDiff` and `Zygote`.

## Constructors

```@docs
VILocationScale
```

```@docs
VIFullRankGaussian
VIMeanFieldGaussian
```

## Gaussian Variational Families

Gaussian variational family:
```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);

L = diagm(ones(2)) |> LowerTriangular;
q = VIFullRankGaussian(μ, L)

L = ones(2) |> Diagonal;
q = VIMeanFieldGaussian(μ, L)
```

## Non-Gaussian Variational Families
Sudent-T Variational Family:

```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);
ν = 3;

# Full-Rank 
L = diagm(ones(2)) |> LowerTriangular;
q = VILocationScale(μ, L, TDist(ν))

# Mean-Field
L = ones(2) |> Diagonal;
q = VILocationScale(μ, L, TDist(ν))
```

Multivariate Laplace family:
```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);

# Full-Rank 
L = diagm(ones(2)) |> LowerTriangular;
q = VILocationScale(μ, L, Laplace())

# Mean-Field
L = ones(2) |> Diagonal;
q = VILocationScale(μ, L, Laplace())
```
