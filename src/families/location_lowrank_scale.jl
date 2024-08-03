
"""
    MvLocationLowRankScale(location, scale_diag, scale_factors, dist) <: ContinuousMultivariateDistribution

Variational family with a covariance in the form of a diagonal matrix plus a squared low-rank matrix.

It generally represents any distribution for which the sampling path can be
represented as follows:
```julia
  d = length(location)
  r = size(scale_factors, 2)
  u_d = rand(dist, d)
  u_f = rand(dist, r)
  z = scale_diag.*u_d + scale_factors*u_f + location
```
"""
struct MvLowRankLocationScale{
    L,
    SD <: AbstractVector,
    SF <: AbstractMatrix,
    D  <: ContinuousDistribution,
    E  <: Real
} <: ContinuousMultivariateDistribution
    location     ::L
    scale_diag   ::SD
    scale_factors::SF
    dist         ::D
    scale_eps    ::E
end

function MvLowRankLocationScale(
    location     ::AbstractVector{T},
    scale_diag   ::AbstractVector{T},
    scale_factors::AbstractMatrix{T},
    dist         ::ContinuousDistribution;
    scale_eps    ::T = sqrt(eps(T))
) where {T <: Real}
    MvLowRankLocationScale(location, scale_diag, scale_factors, dist, scale_eps)
end

Functors.@functor MvLowRankLocationScale (location, scale_diag, scale_factors)

Base.length(q::MvLowRankLocationScale) = length(q.location)

Base.size(q::MvLowRankLocationScale) = size(q.location)

Base.eltype(::Type{<:MvLowRankLocationScale{S, D, L}}) where {S, D, L} = eltype(D)

function StatsBase.entropy(q::MvLowRankLocationScale)
    #@unpack  location, scale, dist = q
    #n_dims = length(location)
    # `convert` is necessary because `entropy` is not type stable upstream
    #n_dims*convert(eltype(location), entropy(dist)) + logdet(scale)
end

function Distributions.logpdf(q::MvLowRankLocationScale, z::AbstractVector{<:Real})
    @unpack location, scale, dist = q
    #sum(Base.Fix1(logpdf, dist), scale \ (z - location)) - logdet(scale)
end

function Distributions.rand(q::MvLowRankLocationScale)
    @unpack location, scale_diag, scale_factors, dist = q
    n_dims    = length(location)
    n_factors = size(scale_factors, 2)
    u_diag    = rand(dist, n_dims)
    u_fact    = rand(dist, n_factors)
    scale_diag.*u_diag + scale_factors*u_fact + location
end

function Distributions.rand(
    rng::AbstractRNG, q::MvLowRankLocationScale{S, D, L}, num_samples::Int
)  where {S, D, L}
    @unpack location, scale_diag, scale_factors, dist = q
    n_dims    = length(location)
    n_factors = size(scale_factors, 2)
    u_diag    = rand(rng, dist, n_dims,    num_samples)
    u_fact    = rand(rng, dist, n_factors, num_samples)
    scale_diag.*u_diag + scale_factors*u_fact .+ location
end

function Distributions._rand!(
    rng::AbstractRNG,
    q  ::MvLowRankLocationScale,
    x  ::AbstractVecOrMat{<:Real}
)
    @unpack location, scale_diag, scale_factors, dist = q

    n_factors = size(scale_factors, 2)

    rand!(rng, dist, x)
    x[:] = scale_diag.*x

    u_fact = rand(dist, n_factors, size(x,2))
    x    .+= scale_factors*u_fact

    return x .+= location
end

Distributions.mean(q::MvLowRankLocationScale) = q.location

function Distributions.var(q::MvLowRankLocationScale)  
    @unpack scale_diag, scale_factors = q
    Diagonal(scale_diag + diag(scale_factors*scale_factors'))
end

function Distributions.cov(q::MvLowRankLocationScale)
    @unpack scale_diag, scale_factors = q
    Diagonal(scale_diag) + scale_factors*scale_factors'
end

function update_variational_params!(
    ::Type{<:MvLowRankLocationScale}, opt_st, params, restructure, grad
)
    opt_st, params = Optimisers.update!(opt_st, params, grad)
    q = restructure(params)
    ϵ = q.scale_eps

    # Project the scale matrix to the set of positive definite triangular matrices
    @. q.scale_diag = max(q.scale_diag, ϵ)

    params, _ = Optimisers.destructure(q)

    opt_st, params
end
