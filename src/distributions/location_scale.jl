
"""

The [location scale] variational family broadly represents various variational
families using `location` and `scale` variational parameters.

Multivariate Student-t variational family with ``\\nu``-degrees of freedom can
be constructed as:
```julia
q₀ = VILocationScale(μ, L, StudentT(ν), eps(Float32))
```

"""
struct VILocationScale{L, S, D, R} <: ContinuousMultivariateDistribution
    location::L
    scale   ::S
    dist    ::D
    epsilon ::R

    function VILocationScale(μ::AbstractVector{<:Real},
                             L::Union{<:AbstractTriangular{<:Real},
                                      <:Diagonal{<:Real}},
                             q_base::ContinuousUnivariateDistribution,
                             epsilon::Real)
        # Restricting all the arguments to have the same types creates problems 
        # with dual-variable-based AD frameworks.
        @assert (length(μ) == size(L,1)) && (length(μ) == size(L,2))
        new{typeof(μ), typeof(L), typeof(q_base), typeof(epsilon)}(μ, L, q_base, epsilon)
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
    mapreduce(zᵢ -> logpdf(dist, zᵢ), +, scale \ (z - location)) - first(logabsdet(scale))
end

function _logpdf(q::VILocationScale, z::AbstractVector{<:Real})
    @unpack location, scale, dist = q
    mapreduce(zᵢ -> logpdf(dist, zᵢ), +, scale \ (z - location)) - first(logabsdet(scale))
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

function VIFullRankGaussian(μ::AbstractVector{T},
                            L::AbstractTriangular{T},
                            epsilon::Real = eps(T)) where {T <: Real}
    q_base = Normal{T}(zero(T), one(T))
    VILocationScale(μ, L, q_base, epsilon)
end

function VIMeanFieldGaussian(μ::AbstractVector{T},
                             L::Diagonal{T},
                             epsilon::Real = eps(T)) where {T <: Real}
    q_base = Normal{T}(zero(T), one(T))
    VILocationScale(μ, L, q_base, epsilon)
end
