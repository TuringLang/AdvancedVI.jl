
"""
    ClosedFormEntropy()

Use closed-form expression of entropy.

# Requirements
- `q` implements `entropy`.

# References
* Titsias, M., & Lázaro-Gredilla, M. (2014, June). Doubly stochastic variational Bayes for non-conjugate inference. In International conference on machine learning (pp. 1971-1979). PMLR.
* Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. Journal of machine learning research.
"""
struct ClosedFormEntropy <: AbstractEntropyEstimator end

maybe_stop_entropy_score(::AbstractEntropyEstimator, q, q_stop) = q

function estimate_entropy(::ClosedFormEntropy, ::Any, q)
    entropy(q)
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

struct MonteCarloEntropy <: AbstractEntropyEstimator end

maybe_stop_entropy_score(::StickingTheLandingEntropy, q, q_stop) = q_stop

function estimate_entropy(
    ::Union{MonteCarloEntropy, StickingTheLandingEntropy},
    mc_samples::AbstractMatrix,
    q
)
    mean(eachcol(mc_samples)) do mc_sample
        -logpdf(q, mc_sample)
    end
end
