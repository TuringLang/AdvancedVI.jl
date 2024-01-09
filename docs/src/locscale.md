
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
  \mathbb{H}(q_{\lambda}) = \mathbb{H}(\varphi) + \log |C|,
```
where ``\mathbb{H}(\varphi)`` is the entropy of the base distribution.
Notice the ``\mathbb{H}(\varphi)`` does not depend on ``\log |C|``.
The derivative of the entropy with respect to ``\lambda`` is thus independent of the base distribution.

## Constructors

!!! note
    For stable convergence, the initial `scale` needs to be sufficiently large and well-conditioned. 
    Initializing `scale` to have small eigenvalues will often result in initial divergences and numerical instabilities.

```@docs
MvLocationScale
```

```@docs
FullRankGaussian
MeanFieldGaussian
```

## Gaussian Variational Families
```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);

L = diagm(ones(2)) |> LowerTriangular;
q = FullRankGaussian(μ, L)

L = ones(2) |> Diagonal;
q = MeanFieldGaussian(μ, L)
```

## Sudent-$$t$$ Variational Families
```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);
ν = 3;

# Full-Rank 
L = diagm(ones(2)) |> LowerTriangular;
q = MvLocationScale(μ, L, TDist(ν))

# Mean-Field
L = ones(2) |> Diagonal;
q = MvLocationScale(μ, L, TDist(ν))
```

## Laplace Variational families
```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);

# Full-Rank 
L = diagm(ones(2)) |> LowerTriangular;
q = MvLocationScale(μ, L, Laplace())

# Mean-Field
L = ones(2) |> Diagonal;
q = MvLocationScale(μ, L, Laplace())
```
