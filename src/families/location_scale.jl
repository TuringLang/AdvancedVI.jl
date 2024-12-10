
"""
    MvLocationScale(location, scale, dist)

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
struct MvLocationScale{S,D<:ContinuousDistribution,L} <: ContinuousMultivariateDistribution
    location::L
    scale::S
    dist::D
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
    return MvLocationScale(location, scale, re.model.dist)
end

function Optimisers.destructure(q::MvLocationScale{<:Diagonal,D,L}) where {D,L}
    (; location, scale, dist) = q
    flat = vcat(location, diag(scale))
    return flat, RestructureMeanField(q)
end
# end

Base.length(q::MvLocationScale) = length(q.location)

Base.size(q::MvLocationScale) = size(q.location)

Base.eltype(::Type{<:MvLocationScale{S,D,L}}) where {S,D,L} = eltype(D)

function StatsBase.entropy(q::MvLocationScale)
    (; location, scale, dist) = q
    n_dims = length(location)
    # `convert` is necessary because `entropy` is not type stable upstream
    return n_dims * convert(eltype(location), entropy(dist)) + logdet(scale)
end

function Distributions.logpdf(q::MvLocationScale, z::AbstractVector{<:Real})
    (; location, scale, dist) = q
    return sum(Base.Fix1(logpdf, dist), scale \ (z - location)) - logdet(scale)
end

function Distributions.rand(q::MvLocationScale)
    (; location, scale, dist) = q
    n_dims = length(location)
    return scale * rand(dist, n_dims) + location
end

function Distributions.rand(
    rng::AbstractRNG, q::MvLocationScale{S,D,L}, num_samples::Int
) where {S,D,L}
    (; location, scale, dist) = q
    n_dims = length(location)
    return scale * rand(rng, dist, n_dims, num_samples) .+ location
end

# This specialization improves AD performance of the sampling path
function Distributions.rand(
    rng::AbstractRNG, q::MvLocationScale{<:Diagonal,D,L}, num_samples::Int
) where {L,D}
    (; location, scale, dist) = q
    n_dims = length(location)
    scale_diag = diag(scale)
    return scale_diag .* rand(rng, dist, n_dims, num_samples) .+ location
end

function Distributions._rand!(
    rng::AbstractRNG, q::MvLocationScale, x::AbstractVecOrMat{<:Real}
)
    (; location, scale, dist) = q
    rand!(rng, dist, x)
    x[:] = scale * x
    return x .+= location
end

function Distributions.mean(q::MvLocationScale)
    (; location, scale) = q
    return location + scale * Fill(mean(q.dist), length(location))
end

function Distributions.var(q::MvLocationScale)
    C = q.scale
    σ2 = var(q.dist)
    return σ2 * diag(C * C')
end

function Distributions.cov(q::MvLocationScale)
    C = q.scale
    σ2 = var(q.dist)
    return σ2 * Hermitian(C * C')
end

"""
    FullRankGaussian(μ, L)

Construct a Gaussian variational approximation with a dense covariance matrix.

# Arguments
- `μ::AbstractVector{T}`: Mean of the Gaussian.
- `L::LinearAlgebra.AbstractTriangular{T}`: Cholesky factor of the covariance of the Gaussian.
"""
function FullRankGaussian(
    μ::AbstractVector{T}, L::LinearAlgebra.AbstractTriangular{T}
) where {T<:Real}
    return MvLocationScale(μ, L, Normal{T}(zero(T), one(T)))
end

"""
    MeanFieldGaussian(μ, L)

Construct a Gaussian variational approximation with a diagonal covariance matrix.

# Arguments
- `μ::AbstractVector{T}`: Mean of the Gaussian.
- `L::Diagonal{T}`: Diagonal Cholesky factor of the covariance of the Gaussian.
"""
function MeanFieldGaussian(μ::AbstractVector{T}, L::Diagonal{T}) where {T<:Real}
    return MvLocationScale(μ, L, Normal{T}(zero(T), one(T)))
end

"""
    ProjectScale(rule, scale_eps)

Compose an optimization `rule` with a projection, where the projection ensures that a `LocationScale` or `LocationScaleLowRank` has a scale with eigenvalues larger than `scale_eps`.

# Arguments
- `rule::Optimisers.AbstractRule`: Optimization rule to compose with the projection.
- `scale_eps::Real`: Lower bound on the eigenvalues of the scale matrix of the projection.
"""
struct ProjectScale{Rule<:Optimisers.AbstractRule,F<:Real} <: Optimisers.AbstractRule
    rule::Rule
    scale_eps::F
end

function ProjectScale(rule, scale_eps::Real=1e-5)
    return ProjectScale{typeof(rule),typeof(scale_eps)}(rule, scale_eps)
end

Optimisers.setup(proj::ProjectScale, x) = Optimisers.setup(proj.rule, x)

Optimisers.init(proj::ProjectScale, x) = Optimisers.init(proj.rule, x)

function update_variational_params!(
    proj::ProjectScale, ::Type{<:MvLocationScale}, opt_st, params, restructure, grad
)
    opt_st, params = Optimisers.update!(opt_st, params, grad)
    q = restructure(params)
    ϵ = convert(eltype(params), proj.scale_eps)

    # Project the scale matrix to the set of positive definite triangular matrices
    diag_idx = diagind(q.scale)
    @. q.scale[diag_idx] = max(q.scale[diag_idx], ϵ)

    params, _ = Optimisers.destructure(q)

    return opt_st, params
end
