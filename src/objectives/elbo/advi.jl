
"""
    ADVI(n_samples; kwargs...)

Automatic differentiation variational inference (ADVI; Kucukelbir *et al.* 2017) objective.
This computes the evidence lower-bound (ELBO) through the ADVI formulation:
```math
\\begin{aligned}
\\mathrm{ADVI}\\left(\\lambda\\right)
&\\triangleq
\\mathbb{E}_{\\eta \\sim q_{\\lambda}}\\left[
  \\log \\pi\\left( \\phi^{-1}\\left( \\eta \\right) \\right)
  +
  \\log \\lvert J_{\\phi^{-1}}\\left(\\eta\\right) \\rvert
\\right]
+ \\mathbb{H}\\left(q_{\\lambda}\\right),
\\end{aligned}
```
where ``\\phi^{-1}`` is an "inverse bijector."

# Arguments
- `n_samples::Int`: Number of Monte Carlo samples used to estimate the ELBO.

# Keyword Arguments
- `entropy`: The estimator for the entropy term. (Type `<: AbstractEntropyEstimator`; Default: ClosedFormEntropy())

# Requirements
- ``q_{\\lambda}`` implements `rand`.
- The target `logdensity(prob, x)` must be differentiable wrt. `x` by the selected AD backend.

Depending on the options, additional requirements on ``q_{\\lambda}`` may apply.

# References
* Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. Journal of machine learning research.
* Titsias, M., & Lázaro-Gredilla, M. (2014, June). Doubly stochastic variational Bayes for non-conjugate inference. In International conference on machine learning (pp. 1971-1979). PMLR.
"""
struct ADVI{EntropyEst <: AbstractEntropyEstimator} <: AbstractVariationalObjective
    entropy  ::EntropyEst
    n_samples::Int
end

ADVI(n_samples::Int; entropy::AbstractEntropyEstimator = ClosedFormEntropy()) = ADVI(entropy, n_samples)

Base.show(io::IO, advi::ADVI) =
    print(io, "ADVI(entropy=$(advi.entropy), n_samples=$(advi.n_samples))")

"""
    estimate_objective_with_samples(obj, prob, q, zs)

Estimate the ELBO using the ADVI formulation over a set of given Monte Carlo samples.

# Arguments
- `advi::ADVI`: ADVI objective.
- `q`: Variational approximation.
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.
- `mc_samples::AbstractMatrix`: Samples to be used to estimate the energy. (Each column is a single sample.)

# Returns
- `obj_est`: Estimate of the objective value.

"""
function estimate_objective_with_samples(
    advi      ::ADVI,
    q         ::Union{Distributions.ContinuousMultivariateDistribution,
                      Bijectors.TransformedDistribution},
    prob,
    mc_samples::AbstractMatrix
)
    estimate_objective_with_samples(advi, q, q, prob, mc_samples)
end


function estimate_objective_with_samples(
    advi      ::ADVI,
    q         ::Distributions.ContinuousMultivariateDistribution,
    q_stop    ::Distributions.ContinuousMultivariateDistribution,
    prob,
    mc_samples::AbstractMatrix
)
    𝔼ℓ = mean(Base.Fix1(LogDensityProblems.logdensity, prob), eachcol(mc_samples))
    ℍ  = estimate_entropy(advi.entropy, mc_samples, q, q_stop)
    𝔼ℓ + ℍ
end

function estimate_objective_with_samples(
    advi        ::ADVI,
    q_trans     ::Bijectors.TransformedDistribution,
    q_trans_stop::Bijectors.TransformedDistribution,
    prob,
    mc_samples_unconstr::AbstractMatrix
)
    @unpack dist, transform = q_trans
    q      = dist
    q_stop = q_trans_stop.dist
    b⁻¹    = transform
    𝔼ℓ     = mean(eachcol(mc_samples_unconstr)) do mc_sample_unconstr
        mc_sample, logdetjacᵢ = Bijectors.with_logabsdet_jacobian(b⁻¹, mc_sample_unconstr)
        LogDensityProblems.logdensity(prob, mc_sample) + logdetjacᵢ
    end
    ℍ  = estimate_entropy(advi.entropy, mc_samples_unconstr, q, q_stop)
    𝔼ℓ + ℍ
end

function rand_uncontrained_samples(
    rng      ::Random.AbstractRNG,
    q        ::ContinuousDistribution,
    n_samples::Int,
)
    rand(rng, q, n_samples)
end

function rand_uncontrained_samples(
    rng      ::Random.AbstractRNG,
    q_trans  ::Bijectors.TransformedDistribution,
    n_samples::Int,
)
    rand(rng, q_trans.dist, n_samples)
end

"""
    estimate_objective([rng,] advi, q, prob; n_samples)

Estimate the ELBO using the ADVI formulation.

# Arguments
- `advi::ADVI`: ADVI objective.
- `q`: Variational approximation
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.

# Keyword Arguments
- `n_samples::Int = advi.n_samples`: Number of samples to be used to estimate the objective.

# Returns
- `obj_est`: Estimate of the objective value.
"""
function estimate_objective(
    rng      ::Random.AbstractRNG,
    advi     ::ADVI,
    q,
    prob;
    n_samples::Int = advi.n_samples
)
    mc_samples_unconstr = rand_uncontrained_samples(rng, q, n_samples)
    estimate_objective_with_samples(advi, q, prob, mc_samples_unconstr)
end

estimate_objective(advi::ADVI, q::Distribution, prob; n_samples::Int = advi.n_samples) =
    estimate_objective(Random.default_rng(), advi, q, prob; n_samples)

function estimate_gradient!(
    rng       ::Random.AbstractRNG,
    advi      ::ADVI,
    adbackend ::ADTypes.AbstractADType,
    out       ::DiffResults.MutableDiffResult,
    prob,
    λ,
    restructure,
    est_state,
)
    q_stop = restructure(λ)
    function f(λ′)
        q = restructure(λ′)
        mc_samples_unconstr = rand_uncontrained_samples(rng, q, advi.n_samples)
        -estimate_objective_with_samples(advi, q, q_stop, prob, mc_samples_unconstr)
    end
    value_and_gradient!(adbackend, f, λ, out)

    nelbo = DiffResults.value(out)
    stat  = (elbo=-nelbo,)

    out, nothing, stat
end
