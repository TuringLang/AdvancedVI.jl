
"""
    VILocationScale(location, scale, dist) <: ContinuousMultivariateDistribution

The location scale variational family broadly represents various variational
families using `location` and `scale` variational parameters.

It generally represents any distribution for which the sampling path can be
represented as follows:
```julia
  d = length(location)
  u = rand(dist, d)
  z = scale*u + location
```

!!! note
    For stable convergence, the initial scale needs to be sufficiently large.
"""
struct VILocationScale{L, S, D} <: ContinuousMultivariateDistribution
    location::L
    scale   ::S
    dist    ::D

    function VILocationScale(location::AbstractVector{<:Real},
                             scale   ::Union{<:AbstractTriangular{<:Real}, <:Diagonal{<:Real}},
                             dist    ::ContinuousUnivariateDistribution)
        # Restricting all the arguments to have the same types creates problems 
        # with dual-variable-based AD frameworks.
        @assert (length(location) == size(scale,1)) && (length(location) == size(scale,2))
        new{typeof(location), typeof(scale), typeof(dist)}(location, scale, dist)
    end
end

Functors.@functor VILocationScale (location, scale)

# Specialization of `Optimisers.destructure` for mean-field location-scale families.
# These are necessary because we only want to extract the diagonal elements of 
# `scale <: Diagonal`, which is not the default behavior. Otherwise, forward-mode AD
# is very inefficient.
# begin
struct RestructureMeanField{L, S<:Diagonal, D}
    q::VILocationScale{L, S, D}
end

function (re::RestructureMeanField)(flat::AbstractVector)
    n_dims   = div(length(flat), 2)
    location = first(flat, n_dims)
    scale    = Diagonal(last(flat, n_dims))
    VILocationScale(location, scale, re.q.dist)
end

function Optimisers.destructure(
    q::VILocationScale{L, <:Diagonal, D}
) where {L, D}
    @unpack location, scale, dist = q
    flat   = vcat(location, diag(scale))
    n_dims = length(location)
    flat, RestructureMeanField(q)
end
# end

Base.length(q::VILocationScale) = length(q.location)
Base.size(q::VILocationScale) = size(q.location)

function StatsBase.entropy(q::VILocationScale)
    @unpack  location, scale, dist = q
    n_dims = length(location)
    n_dims*entropy(dist) + first(logabsdet(scale))
end

function logpdf(q::VILocationScale, z::AbstractVector{<:Real})
    @unpack location, scale, dist = q
    sum(Base.Fix1(logpdf, dist), scale \ (z - location)) - first(logabsdet(scale))
end

function _logpdf(q::VILocationScale, z::AbstractVector{<:Real})
    @unpack location, scale, dist = q
    sum(Base.Fix1(logpdf, dist), scale \ (z - location)) - first(logabsdet(scale))
end

function rand(q::VILocationScale)
    @unpack location, scale, dist = q
    n_dims = length(location)
    scale*rand(dist, n_dims) + location
end

function rand(rng::AbstractRNG, q::VILocationScale, num_samples::Int) 
    @unpack location, scale, dist = q
    n_dims = length(location)
    scale*rand(rng, dist, n_dims, num_samples) .+ location
end

# This specialization improves AD performance of the sampling path
function rand(
    rng::AbstractRNG, q::VILocationScale{L, <:Diagonal, D}, num_samples::Int
) where {L, D}
    @unpack location, scale, dist = q
    n_dims     = length(location)
    scale_diag = diag(scale)
    scale_diag.*rand(rng, dist, n_dims, num_samples) .+ location
end

function _rand!(rng::AbstractRNG, q::VILocationScale, x::AbstractVector{<:Real})
    @unpack location, scale, dist = q
    rand!(rng, dist, x)
    x .= scale*x
    return x += location
end

function _rand!(rng::AbstractRNG, q::VILocationScale, x::AbstractMatrix{<:Real})
    @unpack location, scale, dist = q
    rand!(rng, dist, x)
    x *= scale
    return x += location
end

"""
    VIFullRankGaussian(μ::AbstractVector{T}, L::AbstractTriangular{T}; check_args = true)

This constructs a multivariate Gaussian distribution with a full rank covariance matrix.
"""
function VIFullRankGaussian(
    μ::AbstractVector{T},
    L::AbstractTriangular{T};
    check_args::Bool = true
) where {T <: Real}
    @assert isposdef(L) "Scale must be positive definite"
    if check_args && (minimum(diag(L)) < sqrt(eps(eltype(L))))
        @warn "Initial scale is too small (minimum eigenvalue is $(minimum(diag(L)))). This might result in unstable optimization behavior."
    end
    q_base = Normal{T}(zero(T), one(T))
    VILocationScale(μ, L, q_base)
end

"""
    VIMeanFieldGaussian(μ::AbstractVector{T}, L::Diagonal{T}; check_args = true)

This constructs a multivariate Gaussian distribution with a diagonal covariance matrix.
"""
function VIMeanFieldGaussian(
    μ::AbstractVector{T},
    L::Diagonal{T};
    check_args::Bool = true
) where {T <: Real}
    @assert isposdef(L) "Scale must be positive definite"
    if check_args && (minimum(diag(L)) < sqrt(eps(eltype(L))))
        @warn "Initial scale is too small (minimum eigenvalue is $(minimum(diag(L)))). This might result in unstable optimization behavior."
    end
    q_base = Normal{T}(zero(T), one(T))
    VILocationScale(μ, L, q_base)
end
