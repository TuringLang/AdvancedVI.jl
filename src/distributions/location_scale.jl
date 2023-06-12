
import Base: rand, _rand!

struct LocationScale{ReparamMvDist <: Bijectors.TransformedDistribution} <: ContinuousMultivariateDistribution
    q_trans::ReparamMvDist

    function LocationScale(μ::AbstractVector,
                           L::Union{<: AbstractTriangular,
                                    <: Diagonal},
                           q₀::ContinuousMultivariateDistribution)
        @assert (length(μ) == size(L,1)) && (length(μ) == size(L,2))
        q_trans = transformed(q₀, Bijectors.Shift(μ) ∘ Bijectors.Scale(L))       
        new{typeof(q_trans)}(q_trans)
    end

    function LocationScale(q_trans::Bijectors.TransformedDistribution)
        new{typeof(q_trans)}(q_trans)
    end
end

Functors.@functor LocationScale

Base.length(q::LocationScale) = length(q.q_trans)
Base.size(q::LocationScale) = size(q.q_trans)

function StatsBase.entropy(q::LocationScale)
    q_base = q.q_trans.dist
    scale  = q.q_trans.transform.inner.a
    entropy(q_base) + first(logabsdet(scale))
end


Distributions.logpdf(q::LocationScale, z::AbstractVector) = logpdf(q.q_trans, z)

_logpdf(q::LocationScale, y::AbstractVector) = _logpdf(q.q_trans, y)

rand(q::LocationScale) = rand(q.q_trans)

rand(rng::Random.AbstractRNG, q::LocationScale, num_samples::Int) = rand(rng, q.q_trans, num_samples)

_rand!(rng::Random.AbstractRNG, q::LocationScale, x::AbstractVector{<:Real}) = _rand!(rng, q.q_trans, x)


function FullRankGaussian(μ::AbstractVector{T},
                          L::AbstractTriangular{T,S}) where {T <: Real, S}
    @assert (length(μ) == size(L,1)) && (length(μ) == size(L,2))
    n_dims = length(μ)
    q_base = MvNormal(FillArrays.Zeros{T}(n_dims), PDMats.ScalMat{T}(n_dims, one(T)))
    LocationScale(μ, L, q_base)
end

function MeanFieldGaussian(μ::AbstractVector{T},
                           L::Diagonal{T,V}) where {T <: Real, V}
    @assert (length(μ) == size(L,1))
    n_dims = length(μ)
    q_base = MvNormal(FillArrays.Zeros{T}(n_dims), PDMats.ScalMat{T}(n_dims, one(T)))
    LocationScale(μ, L, q_base)
end
