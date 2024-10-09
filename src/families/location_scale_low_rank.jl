
struct MvLocationScaleLowRank{
    L,SD<:AbstractVector,SF<:AbstractMatrix,D<:ContinuousDistribution,E<:Real
} <: ContinuousMultivariateDistribution
    location::L
    scale_diag::SD
    scale_factors::SF
    dist::D
    scale_eps::E
end

"""
    MvLocationLowRankScale(location, scale_diag, scale_factors, dist; scale_eps)

Variational family with a covariance in the form of a diagonal matrix plus a squared low-rank matrix.
The rank is given by `size(scale_factors, 2)`.

It generally represents any distribution for which the sampling path can be
represented as follows:
```julia
  d = length(location)
  r = size(scale_factors, 2)
  u_diag = rand(dist, d)
  u_factors = rand(dist, r)
  z = scale_diag.*u_diag + scale_factors*u_factors + location
```

`scale_eps` sets a constraint on the smallest value of `scale_diag` to be enforced during optimization.
This is necessary to guarantee stable convergence.

# Keyword Arguments
- `scale_eps`: Lower bound constraint for the values of scale_diag. (default: `sqrt(eps(T))`).
"""
function MvLocationScaleLowRank(
    location::AbstractVector{T},
    scale_diag::AbstractVector{T},
    scale_factors::AbstractMatrix{T},
    dist::ContinuousUnivariateDistribution;
    scale_eps::T=T(1e-4),
) where {T<:Real}
    @assert minimum(scale_diag) ≥ scale_eps "Initial scale is too small (smallest diagonal scale value is $(minimum(scale_diag)). This might result in unstable optimization behavior."
    @assert size(scale_factors, 1) == length(scale_diag)
    return MvLocationScaleLowRank(location, scale_diag, scale_factors, dist, scale_eps)
end

Functors.@functor MvLocationScaleLowRank (location, scale_diag, scale_factors)

Base.length(q::MvLocationScaleLowRank) = length(q.location)

Base.size(q::MvLocationScaleLowRank) = size(q.location)

Base.eltype(::Type{<:MvLocationScaleLowRank{L,SD,SF,D,E}}) where {L,SD,SF,D,E} = eltype(L)

function StatsBase.entropy(q::MvLocationScaleLowRank)
    (; location, scale_diag, scale_factors, dist) = q
    n_dims = length(location)
    scale_diag2 = scale_diag .* scale_diag
    UtDinvU = Hermitian(scale_factors' * (scale_factors ./ scale_diag2))
    logdetΣ = 2 * sum(log.(scale_diag)) + logdet(I + UtDinvU)
    return n_dims * convert(eltype(location), entropy(dist)) + logdetΣ / 2
end

function Distributions.logpdf(
    q::MvLocationScaleLowRank, z::AbstractVector{<:Real}; non_differntiable::Bool=false
)
    (; location, scale_diag, scale_factors, dist) = q
    μ_base = mean(dist)
    n_dims = length(location)

    scale2chol = if non_differntiable
        # Fast O(kd^2) path (not supported by most current AD frameworks):
        scale2chol = Cholesky(LowerTriangular(diagm(sqrt.(scale_diag))))
        n_factors = size(scale_factors, 2)
        for k in 1:n_factors
            factor = scale_factors[:, k] # copy necessary due to in-place mutation
            lowrankupdate!(scale2chol, factor)
        end
        scale2chol
    else
        # Slow but differentiable O(d^3) path
        scale2 = Diagonal(scale_diag .* scale_diag) + scale_factors * scale_factors'
        cholesky(scale2)
    end
    z_std = z - mean(q) + scale2chol.L * Fill(μ_base, n_dims)
    return sum(Base.Fix1(logpdf, dist), scale2chol.L \ z_std) - logdet(scale2chol.L)
end

function Distributions.rand(q::MvLocationScaleLowRank)
    (; location, scale_diag, scale_factors, dist) = q
    n_dims = length(location)
    n_factors = size(scale_factors, 2)
    u_diag = rand(dist, n_dims)
    u_fact = rand(dist, n_factors)
    return scale_diag .* u_diag + scale_factors * u_fact + location
end

function Distributions.rand(
    rng::AbstractRNG, q::MvLocationScaleLowRank{S,D,L}, num_samples::Int
) where {S,D,L}
    (; location, scale_diag, scale_factors, dist) = q
    n_dims = length(location)
    n_factors = size(scale_factors, 2)
    u_diag = rand(rng, dist, n_dims, num_samples)
    u_fact = rand(rng, dist, n_factors, num_samples)
    return scale_diag .* u_diag + scale_factors * u_fact .+ location
end

function Distributions._rand!(
    rng::AbstractRNG, q::MvLocationScaleLowRank, x::AbstractVecOrMat{<:Real}
)
    (; location, scale_diag, scale_factors, dist) = q

    rand!(rng, dist, x)
    x[:] = scale_diag .* x

    u_fact = rand(rng, dist, size(scale_factors, 2), size(x, 2))
    x[:, :] += scale_factors * u_fact

    return x .+= location
end

function Distributions.mean(q::MvLocationScaleLowRank)
    (; location, scale_diag, scale_factors) = q
    μ = mean(q.dist)
    return location +
           scale_diag .* Fill(μ, length(scale_diag)) +
           scale_factors * Fill(μ, size(scale_factors, 2))
end

function Distributions.var(q::MvLocationScaleLowRank)
    (; scale_diag, scale_factors) = q
    σ2 = var(q.dist)
    return σ2 *
           (scale_diag .* scale_diag + sum(scale_factors .* scale_factors; dims=2)[:, 1])
end

function Distributions.cov(q::MvLocationScaleLowRank)
    (; scale_diag, scale_factors) = q
    σ2 = var(q.dist)
    return σ2 * (Diagonal(scale_diag .* scale_diag) + scale_factors * scale_factors')
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

    return opt_st, params
end

"""
    LowRankGaussian(μ, D, U; scale_eps)

Construct a Gaussian variational approximation with a diagonal plus low-rank covariance matrix.

# Arguments
- `μ::AbstractVector{T}`: Mean of the Gaussian.
- `D::Vector{T}`: Diagonal of the scale.
- `U::Matrix{T}`: Low-rank factors of the scale, where `size(U,2)` is the rank.

# Keyword Arguments
- `scale_eps`: Smallest value allowed for the diagonal of the scale. (default: `1e-4`).
"""
function LowRankGaussian(
    μ::AbstractVector{T}, D::Vector{T}, U::Matrix{T}; scale_eps::T=T(1e-4)
) where {T<:Real}
    q_base = Normal{T}(zero(T), one(T))
    return MvLocationScaleLowRank(μ, D, U, q_base; scale_eps)
end
