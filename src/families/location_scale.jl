
"""
    MvLocationScale(location, scale, dist) <: ContinuousMultivariateDistribution

The location scale variational family broadly represents various variational
families using `location` and `scale` variational parameters.

It generally represents any distribution for which the sampling path can be
represented as follows:
```julia
  d = length(location)
  u = rand(dist, d)
  z = scale*u + location
```
"""
struct MvLocationScale{
    S, D <: ContinuousDistribution, L
} <: ContinuousMultivariateDistribution
    location::L
    scale   ::S
    dist    ::D
end

Functors.@functor MvLocationScale (location, scale)

# Specialization of `Optimisers.destructure` for mean-field location-scale families.
# These are necessary because we only want to extract the diagonal elements of 
# `scale <: Diagonal`, which is not the default behavior. Otherwise, forward-mode AD
# is very inefficient.
# begin
struct RestructureMeanField{S <: Diagonal, D, L}
    q::MvLocationScale{S, D, L}
end

function (re::RestructureMeanField)(flat::AbstractVector)
    n_dims   = div(length(flat), 2)
    location = first(flat, n_dims)
    scale    = Diagonal(last(flat, n_dims))
    MvLocationScale(location, scale, re.q.dist)
end

function Optimisers.destructure(
    q::MvLocationScale{<:Diagonal, D, L}
) where {D, L}
    @unpack location, scale, dist = q
    flat   = vcat(location, diag(scale))
    flat, RestructureMeanField(q)
end
# end

Base.length(q::MvLocationScale) = length(q.location)

Base.size(q::MvLocationScale) = size(q.location)

Base.eltype(::Type{<:MvLocationScale{S, D, L}}) where {S, D, L} = eltype(D)

function StatsBase.entropy(q::MvLocationScale)
    @unpack  location, scale, dist = q
    n_dims = length(location)
    n_dims*convert(eltype(location), entropy(dist)) + first(logabsdet(scale))
end

function Distributions.logpdf(q::MvLocationScale, z::AbstractVector{<:Real})
    @unpack location, scale, dist = q
    sum(Base.Fix1(logpdf, dist), scale \ (z - location)) - first(logabsdet(scale))
end

function Distributions._logpdf(q::MvLocationScale, z::AbstractVector{<:Real})
    @unpack location, scale, dist = q
    sum(Base.Fix1(logpdf, dist), scale \ (z - location)) - first(logabsdet(scale))
end

function Distributions.rand(q::MvLocationScale)
    @unpack location, scale, dist = q
    n_dims = length(location)
    scale*rand(dist, n_dims) + location
end

function Distributions.rand(
    rng::AbstractRNG, q::MvLocationScale{S, D, L}, num_samples::Int
)  where {S, D, L}
    @unpack location, scale, dist = q
    n_dims = length(location)
    scale*rand(rng, dist, n_dims, num_samples) .+ location
end

# This specialization improves AD performance of the sampling path
function Distributions.rand(
    rng::AbstractRNG, q::MvLocationScale{<:Diagonal, D, L}, num_samples::Int
) where {L, D}
    @unpack location, scale, dist = q
    n_dims     = length(location)
    scale_diag = diag(scale)
    scale_diag.*rand(rng, dist, n_dims, num_samples) .+ location
end

function Distributions._rand!(rng::AbstractRNG, q::MvLocationScale, x::AbstractVecOrMat{<:Real})
    @unpack location, scale, dist = q
    rand!(rng, dist, x)
    x[:] = scale*x
    return x .+= location
end

Distributions.mean(q::MvLocationScale) = q.location

function Distributions.var(q::MvLocationScale)  
    C = q.scale
    Diagonal(C*C')
end

function Distributions.cov(q::MvLocationScale)
    C = q.scale
    Hermitian(C*C')
end

"""
    FullRankGaussian(location, scale; check_args = true)

Construct a Gaussian variational approximation with a dense covariance matrix.

# Arguments
- `location::AbstractVector{T}`: Mean of the Gaussian.
- `scale::LinearAlgebra.AbstractTriangular{T}`: Cholesky factor of the covariance of the Gaussian.

# Keyword Arguments
- `check_args`: Check the conditioning of the initial scale (default: `true`).
"""
function FullRankGaussian(
    μ::AbstractVector{T},
    L::LinearAlgebra.AbstractTriangular{T};
    check_args::Bool = true
) where {T <: Real}
    @assert minimum(diag(L)) > eps(eltype(L)) "Scale must be positive definite"
    if check_args && (minimum(diag(L)) < sqrt(eps(eltype(L))))
        @warn "Initial scale is too small (minimum eigenvalue is $(minimum(diag(L)))). This might result in unstable optimization behavior."
    end
    q_base = Normal{T}(zero(T), one(T))
    MvLocationScale(μ, L, q_base)
end

"""
    MeanFieldGaussian(location, scale; check_args = true)

Construct a Gaussian variational approximation with a diagonal covariance matrix.

# Arguments
- `location::AbstractVector{T}`: Mean of the Gaussian.
- `scale::Diagonal{T}`: Diagonal Cholesky factor of the covariance of the Gaussian.

# Keyword Arguments
- `check_args`: Check the conditioning of the initial scale (default: `true`).
"""
function MeanFieldGaussian(
    μ::AbstractVector{T},
    L::Diagonal{T};
    check_args::Bool = true
) where {T <: Real}
    @assert minimum(diag(L)) > eps(eltype(L)) "Scale must be a Cholesky factor"
    if check_args && (minimum(diag(L)) < sqrt(eps(eltype(L))))
        @warn "Initial scale is too small (minimum eigenvalue is $(minimum(diag(L)))). This might result in unstable optimization behavior."
    end
    q_base = Normal{T}(zero(T), one(T))
    MvLocationScale(μ, L, q_base)
end
