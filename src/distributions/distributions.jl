## Series of variation of the MvNormal distribution, different methods need different parametrizations ##
abstract type AbstractPosteriorMvNormal{T} <: Distributions.AbstractMvNormal end

Base.length(d::AbstractPosteriorMvNormal) = d.dim
Distributions.dim(d::AbstractPosteriorMvNormal) = d.dim
Distributions.mean(d::AbstractPosteriorMvNormal) = d.μ
rank(d::AbstractPosteriorMvNormal) = d.dim
function eval_entropy(::VariationalInference, d::AbstractPosteriorMvNormal)
    return Distributions.entropy(d)
end
Distributions.logdetcov(d::AbstractPosteriorMvNormal) = logdet(cov(d))
Distributions.invcov(d::AbstractPosteriorMvNormal) = inv(cov(d))
function Distributions.entropy(d::AbstractPosteriorMvNormal)
    return 0.5 * (logdet(cov(d)) + length(d) * log2π)
end

function Distributions._logpdf(d::AbstractPosteriorMvNormal, x::AbstractArray)
    return Distributions._logpdf(MvNormal(d), x)
end

function Distributions._rand!(
    rng::AbstractRNG, d::AbstractPosteriorMvNormal{T}, x::AbstractVector
) where {T}
    return Distributions._rand!(rng, MvNormal(d), x)
end

function Distributions._rand!(
    rng::AbstractRNG, d::AbstractPosteriorMvNormal{T}, x::AbstractMatrix
) where {T}
    return Distributions._rand!(rng, MvNormal(d), x)
end

function Distributions.MvNormal(d::AbstractPosteriorMvNormal)
    return Distributions.MvNormal(mean(d), cov(d))
end

## Update methods

function update_mean!(q::Bijectors.TransformedDistribution, Δ, opt)
    return update_mean!(q.dist, Δ, opt)
end

function update_mean!(q::AbstractPosteriorMvNormal, Δ, opt)
    return q.μ .+= Optimise.apply!(opt, q.μ, Δ)
end

## Flattening and reconstruction methods

function to_vec(q::Bijectors.TransformedDistribution)
    return to_vec(q.dist)
end

function to_dist(q::Bijectors.TransformedDistribution, θ::AbstractVector)
    return transformed(to_dist(q.dist, θ), q.transform)
end

include("cholmvnormal.jl")
include("diagmvnormal.jl")
