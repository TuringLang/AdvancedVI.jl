# [Reparameterizable Variational Families](@id families)

The [RepGradELBO](@ref repgradelbo) objective assumes that the members of the variational family have a differentiable sampling path.
We provide multiple pre-packaged variational families that can be readily used.

## The `LocationScale` Family

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
  q_{\lambda}(z) = {|C|}^{-1} \varphi(C^{-1}(z - m)),
```

the covariance is given as

```math
  \mathrm{Var}\left(q_{\lambda}\right) = C \mathrm{Var}(q_{\lambda}) C^{\top}
```

and the entropy is given as

```math
  \mathbb{H}(q_{\lambda}) = \mathbb{H}(\varphi) + \log |C|,
```

where ``\mathbb{H}(\varphi)`` is the entropy of the base distribution.
Notice the ``\mathbb{H}(\varphi)`` does not depend on ``\log |C|``.
The derivative of the entropy with respect to ``\lambda`` is thus independent of the base distribution.

!!! note
    
    For stable convergence, the initial `scale` needs to be sufficiently large and well-conditioned.
    Initializing `scale` to have small eigenvalues will often result in initial divergences and numerical instabilities.

```@docs
MvLocationScale
```

The following are specialized constructors for convenience:

```@docs
FullRankGaussian
MeanFieldGaussian
```

### Gaussian Variational Families

```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);

L = LowerTriangular(diagm(ones(2)));
q = FullRankGaussian(μ, L)

L = Diagonal(ones(2));
q = MeanFieldGaussian(μ, L)
```

### Student-$$t$$ Variational Families

```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);
ν = 3;

# Full-Rank 
L = LowerTriangular(diagm(ones(2)));
q = MvLocationScale(μ, L, TDist(ν))

# Mean-Field
L = Diagonal(ones(2));
q = MvLocationScale(μ, L, TDist(ν))
```

### Laplace Variational families

```julia
using AdvancedVI, LinearAlgebra, Distributions;
μ = zeros(2);

# Full-Rank 
L = LowerTriangular(diagm(ones(2)));
q = MvLocationScale(μ, L, Laplace())

# Mean-Field
L = Diagonal(ones(2));
q = MvLocationScale(μ, L, Laplace())
```

## The `LocationScaleLowRank` Family

In practice, `LocationScale` families with full-rank scale matrices are known to converge slowly as they require a small SGD stepsize.
Low-rank variational families can be an effective alternative[^ONS2018].
`LocationScaleLowRank` generally represent any ``d``-dimensional distribution which its sampling path can be represented as

```math
z \sim  q_{\lambda} \qquad\Leftrightarrow\qquad
z \stackrel{d}{=} D u_1 + U u_2  + m;\quad u_1, u_2 \sim \varphi
```

where ``D \in \mathbb{R}^{d \times d}`` is a diagonal matrix, ``U \in \mathbb{R}^{d \times r}`` is a dense low-rank matrix for the rank ``r > 0``, ``m \in \mathbb{R}^d`` is the location, and ``\varphi`` is the *base distribution*.
``m``, ``D``, and ``U`` form the variational parameters ``\lambda = (m, D, U)``.

The covariance of this distribution is given as

```math
  \mathrm{Var}\left(q_{\lambda}\right) = D \mathrm{Var}(\varphi) D + U \mathrm{Var}(\varphi) U^{\top}
```

and the entropy is given by the matrix determinant lemma as

```math
  \mathbb{H}(q_{\lambda}) 
  = \mathbb{H}(\varphi) + \log |\Sigma|
  = \mathbb{H}(\varphi) + 2 \log |D| + \log |I + U^{\top} D^{-2} U|,
```

where ``\mathbb{H}(\varphi)`` is the entropy of the base distribution.

!!! note
    
    `logpdf` for `LocationScaleLowRank` is unfortunately not computationally efficient and has the same time complexity as `LocationScale` with a full-rank scale.

```@docs
MvLocationScaleLowRank
```

The following is a specialized constructor for convenience:

```@docs
LowRankGaussian
```

[^ONS2018]: Ong, V. M. H., Nott, D. J., & Smith, M. S. (2018). Gaussian variational approximation with a factor covariance structure. Journal of Computational and Graphical Statistics, 27(3), 465-478.
