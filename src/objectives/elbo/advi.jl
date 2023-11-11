
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

ADVI(
    n_samples::Int;
    entropy  ::AbstractEntropyEstimator = ClosedFormEntropy()
) = ADVI(entropy, n_samples)

Base.show(io::IO, advi::ADVI) =
    print(io, "ADVI(entropy=$(advi.entropy), n_samples=$(advi.n_samples))")

maybe_stop_entropy_score(::StickingTheLandingEntropy, q, q_stop) = q_stop

maybe_stop_entropy_score(::AbstractEntropyEstimator, q, q_stop) = q

function estimate_entropy_maybe_stl(entropy_estimator::AbstractEntropyEstimator, mc_samples, q, q_stop)
    q_maybe_stop = maybe_stop_entropy_score(entropy_estimator, q, q_stop)
    estimate_entropy(entropy_estimator, mc_samples, q_maybe_stop)
end

function estimate_energy_with_samples(::ADVI, mc_samples::AbstractMatrix, prob)
    mean(Base.Fix1(LogDensityProblems.logdensity, prob), eachcol(mc_samples))
end

function estimate_energy_with_samples_bijector(::ADVI, mc_samples::AbstractMatrix, invbij, prob)
    mean(eachcol(mc_samples)) do mc_sample
        mc_sample, logdetjacᵢ = Bijectors.with_logabsdet_jacobian(invbij, mc_sample)
        LogDensityProblems.logdensity(prob, mc_sample) + logdetjacᵢ
    end
end

function estimate_advi_maybe_stl_with_samples(
    advi      ::ADVI,
    q         ::ContinuousDistribution,
    q_stop    ::ContinuousDistribution,
    mc_samples::AbstractMatrix,
    prob
)
    energy  = estimate_energy_with_samples(advi, mc_samples, prob)
    entropy = estimate_entropy_maybe_stl(advi.entropy, mc_samples, q, q_stop)
    energy + entropy
end

function estimate_advi_maybe_stl_with_samples(
    advi        ::ADVI,
    q_trans     ::Bijectors.TransformedDistribution,
    q_trans_stop::Bijectors.TransformedDistribution,
    mc_samples  ::AbstractMatrix,
    prob
)
    q       = q_trans.dist
    invbij  = q_trans.transform
    q_stop  = q_trans_stop.dist
    energy  = estimate_energy_with_samples_bijector(advi, mc_samples, invbij, prob)
    entropy = estimate_entropy_maybe_stl(advi.entropy, mc_samples, q, q_stop)
    energy + entropy
end

rand_unconstrained(
    rng      ::Random.AbstractRNG,
    q        ::ContinuousDistribution,
    n_samples::Int
) = rand(rng, q, n_samples)

rand_unconstrained(
    rng      ::Random.AbstractRNG,
    q        ::Bijectors.TransformedDistribution,
    n_samples::Int
) = rand(rng, q.dist, n_samples)

function estimate_advi_maybe_stl(rng::Random.AbstractRNG, advi::ADVI, q, q_stop, prob)
    mc_samples = rand_unconstrained(rng, q, advi.n_samples)
    estimate_advi_maybe_stl_with_samples(advi, q, q_stop, mc_samples, prob)
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
    mc_samples = rand_unconstrained(rng, q, n_samples)
    estimate_advi_maybe_stl_with_samples(advi, q, q, mc_samples, prob)
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
        elbo = estimate_advi_maybe_stl(rng, advi, q, q_stop, prob)
        -elbo
    end
    value_and_gradient!(adbackend, f, λ, out)

    nelbo = DiffResults.value(out)
    stat  = (elbo=-nelbo,)

    out, nothing, stat
end
