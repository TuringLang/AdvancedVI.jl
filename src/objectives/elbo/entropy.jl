
"""
    ClosedFormEntropy()

Use closed-form expression of entropy[^TL2014][^KTRGB2017].

# Requirements
- The variational approximation implements `entropy`.
"""
struct ClosedFormEntropy <: AbstractEntropyEstimator end

maybe_stop_entropy_score(::AbstractEntropyEstimator, q, q_stop) = q

function estimate_entropy(::ClosedFormEntropy, ::Any, q)
    return entropy(q)
end

"""
    StickingTheLandingEntropy()

The "sticking the landing" entropy estimator[^RWD2017].

# Requirements
- The variational approximation `q` implements `logpdf`.
- `logpdf(q, η)` must be differentiable by the selected AD framework.
"""
struct StickingTheLandingEntropy <: AbstractEntropyEstimator end

struct MonteCarloEntropy <: AbstractEntropyEstimator end

maybe_stop_entropy_score(::StickingTheLandingEntropy, q, q_stop) = q_stop

function estimate_entropy(
    ::Union{MonteCarloEntropy,StickingTheLandingEntropy}, mc_samples::AbstractMatrix, q
)
    mean(eachcol(mc_samples)) do mc_sample
        -logpdf(q, mc_sample)
    end
end
