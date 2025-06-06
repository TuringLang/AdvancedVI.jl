
# ELBO-specific interfaces
abstract type AbstractEntropyEstimator end

"""
    estimate_entropy(entropy_estimator, mc_samples, q, q_stop)

Estimate the entropy of `q`.

# Arguments
- `entropy_estimator`: Entropy estimation strategy.
- `q`: Variational approximation.
- `q_stop`: Variational approximation with detached from the automatic differentiation graph.
- `mc_samples`: Monte Carlo samples used to estimate the entropy. (Only used for Monte Carlo strategies.)

# Returns
- `obj_est`: Estimate of the objective value.
"""
function estimate_entropy end

include("repgradelbo.jl")
include("scoregradelbo.jl")
include("entropy.jl")

export RepGradELBO,
    ScoreGradELBO,
    ClosedFormEntropy,
    StickingTheLandingEntropy,
    MonteCarloEntropy,
    ClosedFormEntropyZeroGradient,
    StickingTheLandingEntropyZeroGradient
