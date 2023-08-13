
struct ClosedFormEntropy <: AbstractEntropyEstimator end

function (::ClosedFormEntropy)(q, ::AbstractMatrix)
    entropy(q)
end

skip_entropy_gradient(::ClosedFormEntropy) = false

abstract type MonteCarloEntropy <: AbstractEntropyEstimator end

struct FullMonteCarloEntropy <: MonteCarloEntropy end

"""
    StickingTheLandingEntropy()

The "sticking the landing" entropy estimator.

# Requirements
- `q` implements `logpdf`.
- `logpdf(q, η)` must be differentiable by the selected AD framework.
"""
struct StickingTheLandingEntropy <: MonteCarloEntropy end

function (::MonteCarloEntropy)(q, ηs::AbstractMatrix)
    n_samples = size(ηs, 2)
    mapreduce(+, eachcol(ηs)) do ηᵢ
        -logpdf(q, ηᵢ) / n_samples
    end
end

