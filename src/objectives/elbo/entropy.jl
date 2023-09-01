
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

# References
* Roeder, G., Wu, Y., & Duvenaud, D. K. (2017). Sticking the landing: Simple, lower-variance gradient estimators for variational inference. Advances in Neural Information Processing Systems, 30.
"""
struct StickingTheLandingEntropy <: AbstractEntropyEstimator end

function (::StickingTheLandingEntropy)(q, ηs::AbstractMatrix)
    @ignore_derivatives mean(eachcol(ηs)) do ηᵢ
        -logpdf(q, ηᵢ)
    end
end
