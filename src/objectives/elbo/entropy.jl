
struct ClosedFormEntropy <: AbstractEntropyEstimator end

function (::ClosedFormEntropy)(q, ::AbstractMatrix)
    entropy(q)
end

struct MonteCarloEntropy <: AbstractEntropyEstimator end

function (::MonteCarloEntropy)(q, ηs::AbstractMatrix)
    mean(eachcol(ηs)) do ηᵢ
        -logpdf(q, ηᵢ)
    end
end

"""
    StickingTheLandingEntropy()

The "sticking the landing" entropy estimator.

# Requirements
- `q` implements `logpdf`.
- `logpdf(q, η)` must be differentiable by the selected AD framework.
"""
struct StickingTheLandingEntropy <: AbstractEntropyEstimator end

function (::StickingTheLandingEntropy)(q, ηs::AbstractMatrix)
    @ignore_derivatives mean(eachcol(ηs)) do ηᵢ
        -logpdf(q, ηᵢ)
    end
end
