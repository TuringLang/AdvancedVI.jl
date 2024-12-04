
"""
	ScoreGradELBO(n_samples; kwargs...)

Evidence lower-bound objective computed with score function gradient with the VarGrad objective, also known as the leave-one-out control variate.

# Arguments
- `n_samples::Int`: Number of Monte Carlo samples used to estimate the VarGrad objective.

# Requirements
- The variational approximation ``q_{\\lambda}`` implements `rand` and `logpdf`.
- `logpdf(q, x)` must be differentiable with respect to `q` by the selected AD backend.
- The target distribution and the variational approximation have the same support.
"""
struct ScoreGradELBO <: AbstractVariationalObjective
    n_samples::Int
end

function Base.show(io::IO, obj::ScoreGradELBO)
    print(io, "ScoreGradELBO(n_samples=")
    print(io, obj.n_samples)
    return print(io, ")")
end

function estimate_objective(
    rng::Random.AbstractRNG, obj::ScoreGradELBO, q, prob; n_samples::Int=obj.n_samples
)
    samples = rand(rng, q, n_samples)
    ℓπ = map(Base.Fix1(LogDensityProblems.logdensity, prob), eachsample(samples))
    ℓq = logpdf.(Ref(q), AdvancedVI.eachsample(samples))
    return mean(ℓπ - ℓq)
end

function estimate_objective(obj::ScoreGradELBO, q, prob; n_samples::Int=obj.n_samples)
    return estimate_objective(Random.default_rng(), obj, q, prob; n_samples)
end

function estimate_scoregradelbo_ad_forward(params′, aux)
    (; logprob, adtype, restructure, samples) = aux
    q = restructure_ad_forward(adtype, restructure, params′)
    ℓπ = logprob
    ℓq = logpdf.(Ref(q), AdvancedVI.eachsample(samples))
    f = ℓq - ℓπ
    return (mean(abs2, f) - mean(f)^2) / 2
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
    q = restructure(params)
    samples = rand(rng, q, obj.n_samples)
    ℓπ = map(Base.Fix1(LogDensityProblems.logdensity, prob), eachsample(samples))
    aux = (adtype=adtype, logprob=ℓπ, restructure=restructure, samples=samples)
    AdvancedVI.value_and_gradient!(
        adtype, estimate_scoregradelbo_ad_forward, params, aux, out
    )
    ℓq = logpdf.(Ref(q), AdvancedVI.eachsample(samples))
    elbo = mean(ℓπ - ℓq)
    stat = (elbo=elbo,)
    return out, nothing, stat
end
