
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
struct MvLocationScale{S,D<:ContinuousDistribution,L,E<:Real} <:
       ContinuousMultivariateDistribution
    location::L
    scale::S
    dist::D
    scale_eps::E
end

function MvLocationScale(
    location::AbstractVector{T},
    scale::AbstractMatrix{T},
    dist::ContinuousDistribution;
    scale_eps::T=sqrt(eps(T)),
) where {T<:Real}
    return MvLocationScale(location, scale, dist, scale_eps)
end

Functors.@functor MvLocationScale (location, scale)

# Specialization of `Optimisers.destructure` for mean-field location-scale families.
# These are necessary because we only want to extract the diagonal elements of 
# `scale <: Diagonal`, which is not the default behavior. Otherwise, forward-mode AD
# is very inefficient.
# begin
struct RestructureMeanField{S<:Diagonal,D,L}
    model::MvLocationScale{S,D,L}
end

function (re::RestructureMeanField)(flat::AbstractVector)
    n_dims = div(length(flat), 2)
    location = first(flat, n_dims)
    scale = Diagonal(last(flat, n_dims))
    return MvLocationScale(location, scale, re.model.dist, re.model.scale_eps)
end

function Optimisers.destructure(q::MvLocationScale{<:Diagonal,D,L}) where {D,L}
    @unpack location, scale, dist = q
    flat = vcat(location, diag(scale))
    return flat, RestructureMeanField(q)
end
# end

Base.length(q::MvLocationScale) = length(q.location)

Base.size(q::MvLocationScale) = size(q.location)

Base.eltype(::Type{<:MvLocationScale{S,D,L}}) where {S,D,L} = eltype(D)

function StatsBase.entropy(q::MvLocationScale)
    @unpack location, scale, dist = q
    n_dims = length(location)
    # `convert` is necessary because `entropy` is not type stable upstream
    return n_dims * convert(eltype(location), entropy(dist)) + logdet(scale)
end

function Distributions.logpdf(q::MvLocationScale, z::AbstractVector{<:Real})
    @unpack location, scale, dist = q
    return sum(Base.Fix1(logpdf, dist), scale \ (z - location)) - logdet(scale)
end

function Distributions.rand(q::MvLocationScale)
    @unpack location, scale, dist = q
    n_dims = length(location)
    return scale * rand(dist, n_dims) + location
end

function Distributions.rand(
    rng::AbstractRNG, q::MvLocationScale{S,D,L}, num_samples::Int
) where {S,D,L}
    @unpack location, scale, dist = q
    n_dims = length(location)
    return scale * rand(rng, dist, n_dims, num_samples) .+ location
end

# This specialization improves AD performance of the sampling path
function Distributions.rand(
    rng::AbstractRNG, q::MvLocationScale{<:Diagonal,D,L}, num_samples::Int
) where {L,D}
    @unpack location, scale, dist = q
    n_dims = length(location)
    scale_diag = diag(scale)
    return scale_diag .* rand(rng, dist, n_dims, num_samples) .+ location
end

function Distributions._rand!(
    rng::AbstractRNG, q::MvLocationScale, x::AbstractVecOrMat{<:Real}
)
    @unpack location, scale, dist = q
    rand!(rng, dist, x)
    x[:] = scale * x
    return x .+= location
end

Distributions.mean(q::MvLocationScale) = q.location

function Distributions.var(q::MvLocationScale)
    C = q.scale
    return Diagonal(C * C')
end

function Distributions.cov(q::MvLocationScale)
    C = q.scale
    return Hermitian(C * C')
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
    μ::AbstractVector{T}, L::LinearAlgebra.AbstractTriangular{T}; scale_eps::T=sqrt(eps(T))
) where {T<:Real}
    @assert minimum(diag(L)) ≥ sqrt(scale_eps) "Initial scale is too small (smallest diagonal value is $(minimum(diag(L)))). This might result in unstable optimization behavior."
    q_base = Normal{T}(zero(T), one(T))
    return MvLocationScale(μ, L, q_base, scale_eps)
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
    μ::AbstractVector{T}, L::Diagonal{T}; scale_eps::T=sqrt(eps(T))
) where {T<:Real}
    @assert minimum(diag(L)) ≥ sqrt(eps(eltype(L))) "Initial scale is too small (smallest diagonal value is $(minimum(diag(L)))). This might result in unstable optimization behavior."
    q_base = Normal{T}(zero(T), one(T))
    return MvLocationScale(μ, L, q_base, scale_eps)
end

function update_variational_params!(
    ::Type{<:MvLocationScale}, opt_st, params, restructure, grad
)
    opt_st, params = Optimisers.update!(opt_st, params, grad)
    q = restructure(params)
    ϵ = q.scale_eps

    # Project the scale matrix to the set of positive definite triangular matrices
    diag_idx = diagind(q.scale)
    @. q.scale[diag_idx] = max(q.scale[diag_idx], ϵ)

    params, _ = Optimisers.destructure(q)

    return opt_st, params
end
