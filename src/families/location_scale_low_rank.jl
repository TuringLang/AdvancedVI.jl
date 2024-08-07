
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
struct MvLocationScaleLowRank{
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

function MvLocationScaleLowRank(
    location     ::AbstractVector{T},
    scale_diag   ::AbstractVector{T},
    scale_factors::AbstractMatrix{T},
    dist         ::ContinuousDistribution;
    scale_eps    ::T = sqrt(eps(T))
) where {T <: Real}
    @assert size(scale_factors,1) == length(scale_diag)
    MvLocationScaleLowRank(location, scale_diag, scale_factors, dist, scale_eps)
end

Functors.@functor MvLocationScaleLowRank (location, scale_diag, scale_factors)

Base.length(q::MvLocationScaleLowRank) = length(q.location)

Base.size(q::MvLocationScaleLowRank) = size(q.location)

Base.eltype(::Type{<:MvLocationScaleLowRank{S, D, L}}) where {S, D, L} = eltype(D)

function StatsBase.entropy(q::MvLocationScaleLowRank)
    @unpack location, scale_diag, scale_factors, dist = q
    n_dims  = length(location)
    scale_diag2 = scale_diag.*scale_diag
    UtDinvU = Hermitian(scale_factors'*(scale_factors./scale_diag2))
    logdetΣ = 2*sum(log.(scale_diag)) + logdet(I + UtDinvU)
    n_dims*convert(eltype(location), entropy(dist)) + logdetΣ/2
end

function Distributions.logpdf(q::MvLocationScaleLowRank, z::AbstractVector{<:Real})
    @unpack location, scale_diag, scale_factors, dist = q
    #
    ## More efficient O(kd^2) but non-differentiable version:
    #
    # Σchol = Cholesky(LowerTriangular(diagm(sqrt.(scale_diag))))
    # n_factors = size(scale_factors, 2)
    # for k in 1:n_factors
    #     factor = scale_factors[:,k]
    #     lowrankupdate!(Σchol, factor)
    # end

    Σ = Diagonal(scale_diag) + scale_factors*scale_factors'
    Σchol = cholesky(Σ)
    sum(Base.Fix1(logpdf, dist), Σchol.L \ (z - location)) - logdet(Σchol.L)
end

function Distributions.rand(q::MvLocationScaleLowRank)
    @unpack location, scale_diag, scale_factors, dist = q
    n_dims    = length(location)
    n_factors = size(scale_factors, 2)
    u_diag    = rand(dist, n_dims)
    u_fact    = rand(dist, n_factors)
    scale_diag.*u_diag + scale_factors*u_fact + location
end

function Distributions.rand(
    rng::AbstractRNG, q::MvLocationScaleLowRank{S, D, L}, num_samples::Int
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
    q  ::MvLocationScaleLowRank,
    x  ::AbstractVecOrMat{<:Real}
)
    @unpack location, scale_diag, scale_factors, dist = q

    rand!(rng, dist, x)
    x[:] = scale_diag.*x

    u_fact = rand(rng, dist, size(scale_factors, 2), size(x,2))
    x[:,:] += scale_factors*u_fact

    return x .+= location
end

Distributions.mean(q::MvLocationScaleLowRank) = q.location

function Distributions.var(q::MvLocationScaleLowRank)  
    @unpack scale_diag, scale_factors = q
    Diagonal(scale_diag.^2 + sum(scale_factors.^2, dims=2)[:,1])
end

function Distributions.cov(q::MvLocationScaleLowRank)
    @unpack scale_diag, scale_factors = q
    Diagonal(scale_diag.^2) + scale_factors*scale_factors'
end

function update_variational_params!(
    ::Type{<:MvLocationScaleLowRank}, opt_st, params, restructure, grad
)
    opt_st, params = Optimisers.update!(opt_st, params, grad)
    q = restructure(params)
    ϵ = q.scale_eps

    # Clip diagonal to guarantee positive definite covariance
    @. q.scale_diag = max(q.scale_diag, ϵ)

    params, _ = Optimisers.destructure(q)

    opt_st, params
end
