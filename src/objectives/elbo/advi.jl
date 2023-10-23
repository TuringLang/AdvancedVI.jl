
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
    (advi::ADVI)(
        [rng], prob, q, zs::AbstractMatrix
    )

Estimate the ELBO of the variational approximation `q` of the target `prob` using the ADVI formulation over the Monte Carlo samples `zs` (each column is a sample).
"""
function (advi::ADVI)(
    prob,
    q ::Distributions.ContinuousMultivariateDistribution,
    zs::AbstractMatrix
)
    𝔼ℓ = mean(Base.Fix1(LogDensityProblems.logdensity, prob), eachcol(zs))
    ℍ  = advi.entropy(q, zs)
    𝔼ℓ + ℍ
end

function (advi::ADVI)(
    prob,
    q_trans::Bijectors.TransformedDistribution,
    ηs     ::AbstractMatrix
)
    @unpack dist, transform = q_trans
    q   = dist
    b⁻¹ = transform
    𝔼ℓ = mean(eachcol(ηs)) do ηᵢ
        zᵢ, logdetjacᵢ = Bijectors.with_logabsdet_jacobian(b⁻¹, ηᵢ)
        LogDensityProblems.logdensity(prob, zᵢ) + logdetjacᵢ
    end
    ℍ  = advi.entropy(q, ηs)
    𝔼ℓ + ℍ
end

"""
    (advi::ADVI)(
        [rng], prob, q; n_samples::Int = advi.n_samples
    )

Estimate the ELBO of the variational approximation `q` of the target `prob` using the ADVI formulation using `n_samples` number of Monte Carlo samples.
"""
function (advi::ADVI)(
    rng      ::Random.AbstractRNG,
    prob,
    q        ::ContinuousDistribution;
    n_samples::Int = advi.n_samples
)
    zs = rand(rng, q, n_samples)
    advi(prob, q, zs)
end

function (advi::ADVI)(
    rng      ::Random.AbstractRNG,
    prob,
    q_trans  ::Bijectors.TransformedDistribution;
    n_samples::Int  = advi.n_samples
)
    q  = q_trans.dist
    ηs = rand(rng, q, n_samples)
    advi(prob, q_trans, ηs)
end

(advi::ADVI)(prob, q::Distribution; n_samples::Int = advi.n_samples) =
    advi(Random.default_rng(), prob, q; n_samples)

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
    function f(λ′)
        q_trans = restructure(λ′)
        q       = q_trans.dist
        ηs      = rand(rng, q, advi.n_samples)
        -advi(prob, q_trans, ηs)
    end
    value_and_gradient!(adbackend, f, λ, out)

    nelbo = DiffResults.value(out)
    stat  = (elbo=-nelbo,)

    out, nothing, stat
end
