
"""
	ScoreGradELBO(n_samples; kwargs...)

Evidence lower-bound objective computed with score function gradients. 
```math
\\begin{aligned}
\\nabla_{\\lambda} \\mathrm{ELBO}\\left(\\lambda\\right)
&\\=
\\mathbb{E}_{z \\sim q_{\\lambda}}\\left[
  \\log \\pi\\left(z\\right) \\nabla_{\\lambda} \\log q_{\\lambda}(z)
\\right]
+ \\mathbb{H}\\left(q_{\\lambda}\\right),
\\end{aligned}
```

To reduce the variance of the gradient estimator, we use a baseline computed from a running average of the previous ELBO values and subtract it from the objective.

```math
\\mathbb{E}_{z \\sim q_{\\lambda}}\\left[
  \\nabla_{\\lambda} \\log q_{\\lambda}(z) \\left(\\pi\\left(z\\right) - \\beta\\right)
\\right]
```

# Arguments
- `n_samples::Int`: Number of Monte Carlo samples used to estimate the ELBO.

# Keyword Arguments
- `entropy`: The estimator for the entropy term. (Type `<: AbstractEntropyEstimator`; Default: `FullMonteCarloEntropy()`)
- `baseline_window_size::Int`: The window size to use to compute the baseline. (Default: `10`)

# Requirements
- The variational approximation ``q_{\\lambda}`` implements `rand` and `logpdf`.
- `logpdf(q, x)` must be differentiable with respect to `q` by the selected AD backend.
- The target distribution and the variational approximation have the same support.

Depending on the options, additional requirements on ``q_{\\lambda}`` may apply.
"""
struct ScoreGradELBO{EntropyEst<:AbstractEntropyEstimator} <:
       AdvancedVI.AbstractVariationalObjective
    entropy::EntropyEst
    n_samples::Int
    baseline_window_size::Int
end

function ScoreGradELBO(
    n_samples::Int;
    entropy::AbstractEntropyEstimator=MonteCarloEntropy(),
    baseline_window_size::Int=10,
)
    return ScoreGradELBO(entropy, n_samples, baseline_window_size)
end

function init(
    ::Random.AbstractRNG, obj::ScoreGradELBO, prob, params::AbstractVector{T}, restructure
) where {T<:Real}
    return MovingWindow(T, obj.baseline_window_size)
end

function Base.show(io::IO, obj::ScoreGradELBO)
    print(io, "ScoreGradELBO(entropy=")
    print(io, obj.entropy)
    print(io, ", n_samples=")
    print(io, obj.n_samples)
    print(io, ", baseline_window_size=")
    print(io, obj.baseline_window_size)
    return print(io, ")")
end

function estimate_energy_with_samples(
    prob, samples_stop, samples_logprob, samples_logprob_stop, baseline
)
    fv = Base.Fix1(LogDensityProblems.logdensity, prob).(eachsample(samples_stop))
    fv_mean = mean(fv)
    score_grad = mean(@. samples_logprob * (fv - baseline))
    score_grad_stop = mean(@. samples_logprob_stop * (fv - baseline))
    return fv_mean + (score_grad - score_grad_stop)
end

function estimate_objective(
    rng::Random.AbstractRNG, obj::ScoreGradELBO, q, prob; n_samples::Int=obj.n_samples
)
    samples, entropy = reparam_with_entropy(rng, q, q, obj.n_samples, obj.entropy)
    energy = map(Base.Fix1(LogDensityProblems.logdensity, prob), eachsample(samples))
    return mean(energy) + entropy
end

function estimate_objective(obj::ScoreGradELBO, q, prob; n_samples::Int=obj.n_samples)
    return estimate_objective(Random.default_rng(), obj, q, prob; n_samples)
end

function estimate_scoregradelbo_ad_forward(params′, aux)
    @unpack rng, obj, problem, adtype, restructure, q_stop, baseline = aux
    q = restructure_ad_forward(adtype, restructure, params′)
    samples_stop = rand(rng, q_stop, obj.n_samples)
    entropy = estimate_entropy_maybe_stl(obj.entropy, samples_stop, q, q_stop)
    samples_logprob = logpdf.(Ref(q), AdvancedVI.eachsample(samples_stop))
    samples_logprob_stop = logpdf.(Ref(q_stop), AdvancedVI.eachsample(samples_stop))
    energy = estimate_energy_with_samples(
        problem, samples_stop, samples_logprob, samples_logprob_stop, baseline
    )
    elbo = energy + entropy
    return -elbo
end

function AdvancedVI.estimate_gradient!(
    rng::Random.AbstractRNG,
    obj::ScoreGradELBO,
    adtype::ADTypes.AbstractADType,
    out::DiffResults.MutableDiffResult,
    prob,
    params,
    restructure,
    state,
)
    baseline_buf = state
    baseline_history = OnlineStats.value(baseline_buf)
    baseline = if isempty(baseline_history)
        one(eltype(params))
    else
        mean(baseline_history)
    end
    q_stop = restructure(params)
    aux = (
        rng=rng,
        adtype=adtype,
        obj=obj,
        problem=prob,
        restructure=restructure,
        baseline=baseline,
        q_stop=q_stop,
    )
    AdvancedVI.value_and_gradient!(
        adtype, estimate_scoregradelbo_ad_forward, params, aux, out
    )
    nelbo = DiffResults.value(out)
    stat = (elbo=-nelbo,)
    if obj.baseline_window_size > 0
        fit!(baseline_buf, -nelbo)
    end
    return out, baseline_buf, stat
end
