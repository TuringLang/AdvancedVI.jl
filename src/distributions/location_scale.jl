
"""
    VILocationScale(location, scale, dist) <: ContinuousMultivariateDistribution

The location scale variational family broadly represents various variational
families using `location` and `scale` variational parameters.

It generally represents any distribution for which the sampling path can be
represented as the following:
```julia
  d = length(location)
  u = rand(dist, d)
  z = scale*u + location
```
"""
struct VILocationScale{L, S, D} <: ContinuousMultivariateDistribution
    location::L
    scale   ::S
    dist    ::D

    function VILocationScale(location::AbstractVector{<:Real},
                             scale::Union{<:AbstractTriangular{<:Real},
                                      <:Diagonal{<:Real}},
                             dist::ContinuousUnivariateDistribution)
        # Restricting all the arguments to have the same types creates problems 
        # with dual-variable-based AD frameworks.
        @assert (length(location) == size(scale,1)) && (length(location) == size(scale,2))
        new{typeof(location), typeof(scale), typeof(dist)}(location, scale, dist)
    end
end

Functors.@functor VILocationScale (location, scale)

Base.length(q::VILocationScale) = length(q.location)
Base.size(q::VILocationScale) = size(q.location)

function StatsBase.entropy(q::VILocationScale)
    @unpack  location, scale, dist = q
    n_dims = length(location)
    n_dims*entropy(dist) + first(logabsdet(scale))
end

function logpdf(q::VILocationScale, z::AbstractVector{<:Real})
    @unpack location, scale, dist = q
    sum(zᵢ -> logpdf(dist, zᵢ), scale \ (z - location)) - first(logabsdet(scale))
end

function _logpdf(q::VILocationScale, z::AbstractVector{<:Real})
    @unpack location, scale, dist = q
    sum(zᵢ -> logpdf(dist, zᵢ), scale \ (z - location)) - first(logabsdet(scale))
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
    VIFullRankGaussian(μ::AbstractVector{T}, L::AbstractTriangular{T})

This constructs a multivariate Gaussian distribution with a full rank covariance matrix.
"""
function VIFullRankGaussian(μ::AbstractVector{T}, L::AbstractTriangular{T}) where {T <: Real}
    q_base = Normal{T}(zero(T), one(T))
    VILocationScale(μ, L, q_base)
end

"""
    VIMeanFieldGaussian(μ::AbstractVector{T}, L::Diagonal{T})

This constructs a multivariate Gaussian distribution with a diagonal covariance matrix.
"""
function VIMeanFieldGaussian(μ::AbstractVector{T}, L::Diagonal{T}) where {T <: Real}
    q_base = Normal{T}(zero(T), one(T))
    VILocationScale(μ, L, q_base)
end
