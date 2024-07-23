"""
    ScoreGradELBO(n_samples; kwargs...)

Evidence lower-bound objective computed with score function gradients.
```math
\\begin{aligned}
\\nabla_{\\lambda}\\mathbb{E}_{z \\sim q_{\\lambda}}\\left[
  \\log \\pi\\left(z\\right)
\\right] = \\mathbb{E}_{z \\sim q_{\\lambda}}\\left[
  \\nabla_{\\lambda} \\log q_{\\lambda}(z) \\pi\\left(z\\right)
\\end{aligned}
```

To reduce the variance of the gradient estimator, we substract ``\\mathbb{E}_{z \\sim q_{\\lambda}}\\left[
  \\pi\\left(z\\right)\right]`` since it is a constant with respect to ``\\lambda``.

# Arguments
- `n_samples::Int`: Number of Monte Carlo samples used to estimate the ELBO.

# Keyword Arguments
- `entropy`: The estimator for the entropy term. (Type `<: AbstractEntropyEstimator`; Default: `ClosedFormEntropy()`)

# Requirements
- The variational approximation ``q_{\\lambda}`` implements `rand` and `logpdf`.
- `logpdf(q, x)` must be differentiable with respect to `q` by the selected AD backend.
- The target distribution and the variational approximation have the same support.

Depending on the options, additional requirements on ``q_{\\lambda}`` may apply.
"""
struct ScoreGradELBO{EntropyEst <: AbstractEntropyEstimator} <:
       AdvancedVI.AbstractVariationalObjective
    entropy::EntropyEst
    n_samples::Int
end

function ScoreGradELBO(
        n_samples::Int;
        entropy::AbstractEntropyEstimator = ClosedFormEntropy()
)
    ScoreGradELBO(entropy, n_samples)
end
function Base.show(io::IO, obj::ScoreGradELBO)
    print(io, "ScoreGradELBO(entropy=")
    print(io, obj.entropy)
    print(io, ", n_samples=")
    print(io, obj.n_samples)
    print(io, ")")
end

function estimate_energy_with_samples(prob, samples, samples_logprob, adtype)
    fv = Base.Fix1(LogDensityProblems.logdensity, prob).(eachsample(samples))
    fv_mean = mean(fv)
    score_grad = mean(samples_logprob .* fv) 
    # this trick is to keep the ELBO the same value and use the score in the gradient pass.
    return stop_gradient(adtype, fv_mean) + (score_grad - stop_gradient(adtype, score_grad))
end

function compute_elbo(q, samples, entropy, problem, adtype)
    samples_nograd = stop_gradient(adtype, samples)
    samples_logprob = logpdf.(Ref(q), AdvancedVI.eachsample(samples_nograd)) 
    energy = estimate_energy_with_samples(problem, samples_nograd, samples_logprob, adtype)
    elbo = energy + entropy
    return elbo
end

function estimate_scoregradelbo_ad_forward(params′, aux)
    @unpack rng, obj, problem, restructure, q_stop, adtype = aux
    q = restructure(params′)
    samples, entropy = reparam_with_entropy(
        rng, q, q_stop, obj.n_samples, obj.entropy)
    elbo = compute_elbo(q, samples, entropy, problem, adtype)
    return -elbo
end

function estimate_objective(
        rng::Random.AbstractRNG,
        obj::ScoreGradELBO,
        q,
        prob;
        n_samples::Int = obj.n_samples
)
    samples, entropy = reparam_with_entropy(rng, q, q, n_samples, obj.entropy)
    energy = Base.Fix1(LogDensityProblems.logdensity, prob).(eachsample(samples))
    return mean(energy) + entropy
end

function estimate_objective(
        obj::ScoreGradELBO, q, prob; n_samples::Int = obj.n_samples)
    estimate_objective(Random.default_rng(), obj, q, prob; n_samples)
end

function AdvancedVI.estimate_gradient!(
        rng::Random.AbstractRNG,
        obj::ScoreGradELBO,
        adtype::ADTypes.AbstractADType,
        out::DiffResults.MutableDiffResult,
        prob,
        params,
        restructure,
        state
)
    q_stop = restructure(params)
    aux = (rng = rng, obj = obj, problem = prob, restructure = restructure, q_stop = q_stop, adtype = adtype)
    AdvancedVI.value_and_gradient!(
        adtype, estimate_scoregradelbo_ad_forward, params, aux, out
    )
    nelbo = DiffResults.value(out)
    stat = (elbo = -nelbo,)
    out, nothing, stat
end
