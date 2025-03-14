
"""
    ClosedFormEntropyZeroGradient()

Use closed-form expression of entropy but detach it from the AD graph.

# Requirements
- The variational approximation implements `entropy`.
"""
struct ClosedFormEntropyZeroGradient <: AbstractEntropyEstimator end

function estimate_entropy(::ClosedFormEntropyZeroGradient, ::Any, ::Any, q_stop)
    return entropy(q_stop)
end

"""
    ClosedFormEntropy()

Use closed-form expression of entropy[^TL2014][^KTRGB2017].

# Requirements
- The variational approximation implements `entropy`.
"""
struct ClosedFormEntropy <: AbstractEntropyEstimator end

function estimate_entropy(::ClosedFormEntropy, ::Any, q, q_stop)
    return entropy(q)
end

"""
    MonteCarloEntropy()

Monte Carlo estimation of the entropy.

# Requirements
- The variational approximation `q` implements `logpdf`.
- `logpdf(q, η)` must be differentiable by the selected AD framework.
"""
struct MonteCarloEntropy <: AbstractEntropyEstimator end

function estimate_entropy(::MonteCarloEntropy, mc_samples::AbstractMatrix, q, q_stop)
    return mean(eachcol(mc_samples)) do mc_sample
        -logpdf(q, mc_sample)
    end
end

"""
    StickingTheLandingEntropy()

The "sticking the landing" entropy estimator[^RWD2017].

# Requirements
- The variational approximation `q` implements `logpdf`.
- `logpdf(q, η)` must be differentiable by the selected AD framework.
"""
struct StickingTheLandingEntropy <: AbstractEntropyEstimator end

function estimate_entropy(
    ::StickingTheLandingEntropy,
    mc_samples::AbstractMatrix,
    q,
    q_stop,
)
    return mean(eachcol(mc_samples)) do mc_sample
        -logpdf(q_stop, mc_sample)
    end
end

"""
    StickingTheLandingEntropyZeroGradient()

# Requirements
- The variational approximation `q` implements `logpdf`.
- `logpdf(q, η)` must be differentiable by the selected AD framework.
- The variational approximation implements `entropy`.
"""
struct StickingTheLandingEntropyZeroGradient <: AbstractEntropyEstimator end

function estimate_entropy(
    ::Union{MonteCarloEntropy,StickingTheLandingEntropyZeroGradient},
    mc_samples::AbstractMatrix,
    q,
    q_stop,
)
    entropy_stl = mean(eachcol(mc_samples)) do mc_sample
        -logpdf(q_stop, mc_sample)
    end
    return entropy_stl - entropy(q) + entropy(q_stop)
end
