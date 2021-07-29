## Series of variation of the MvNormal distribution, different methods need different parametrizations ##
abstract type AbstractPosteriorMvNormal{T} <:
              Distributions.AbstractMvNormal end

Base.length(d::AbstractPosteriorMvNormal) = d.dim
Distributions.dim(d::AbstractPosteriorMvNormal) = d.dim
Distributions.mean(d::AbstractPosteriorMvNormal) = d.μ
rank(d::AbstractPosteriorMvNormal) = d.dim
eval_entropy(::VariationalInference, d::AbstractPosteriorMvNormal) = Distributions.entropy(d)
Distributions.logdetcov(d::AbstractPosteriorMvNormal) = logdet(cov(d))
Distributions.invcov(d::AbstractPosteriorMvNormal) = inv(cov(d))
Distributions.entropy(d::AbstractPosteriorMvNormal) = 0.5 * (logdet(cov(d)) + length(d) * log2π)

function Distributions._logpdf(d::AbstractPosteriorMvNormal, x::AbstractArray)
    Distributions._logpdf(MvNormal(d), x)
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractPosteriorMvNormal{T},
  x::AbstractVector,
) where {T}
    Distributions._rand!(rng, MvNormal(d), x)
end

function Distributions._rand!(
  rng::AbstractRNG,
  d::AbstractPosteriorMvNormal{T},
  x::AbstractMatrix,
) where {T}
    Distributions._rand!(rng, MvNormal(d), x)
end

Distributions.MvNormal(d::AbstractPosteriorMvNormal) = Distributions.MvNormal(mean(d), cov(d))

## Update methods

function update_mean!(q::Bijectors.TransformedDistribution, Δ, opt)
    return update_mean!(q.dist, Δ, opt)
end

function update_mean!(q::AbstractPosteriorMvNormal, Δ, opt)
    return q.μ .+= Optimise.apply!(opt, q.μ, Δ)
end

## Flattening and reconstruction methods

function to_vec(q::Bijectors.TransformedDistribution)
    to_vec(q.dist)
end


function to_dist(q::Bijectors.TransformedDistribution, θ::AbstractVector)
    transformed(to_dist(q.dist, θ), q.transform)
end


include("cholmvnormal.jl")
include("diagmvnormal.jl")