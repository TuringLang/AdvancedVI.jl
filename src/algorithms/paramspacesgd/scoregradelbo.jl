
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

function init(
    rng::Random.AbstractRNG,
    obj::ScoreGradELBO,
    adtype::ADTypes.AbstractADType,
    prob,
    params,
    restructure,
)
    q = restructure(params)
    samples = rand(rng, q, obj.n_samples)
    ℓπ = map(Base.Fix1(LogDensityProblems.logdensity, prob), eachsample(samples))
    aux = (adtype=adtype, logprob_stop=ℓπ, samples_stop=samples, restructure=restructure)
    obj_ad_prep = AdvancedVI._prepare_gradient(
        estimate_scoregradelbo_ad_forward, adtype, params, aux
    )
    return (obj_ad_prep=obj_ad_prep, problem=prob)
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

"""
    estimate_scoregradelbo_ad_forward(params, aux)

AD-guaranteed forward path of the score gradient objective.

# Arguments
- `params`: Variational parameters.
- `aux`: Auxiliary information excluded from the AD path.

# Auxiliary Information 
`aux` should containt the following entries:
- `samples_stop`: Samples drawn from `q = restructure(params)` but with their gradients stopped (excluded from the AD path).
- `logprob_stop`: Log-densities of the target `LogDensityProblem` evaluated over `samples_stop`.
- `adtype`: The `ADType` used for differentiating the forward path.
- `restructure`: Callable for restructuring the varitional distribution from `params`.
"""
function estimate_scoregradelbo_ad_forward(params, aux)
    (; samples_stop, logprob_stop, adtype, restructure) = aux
    q = restructure_ad_forward(adtype, restructure, params)
    ℓπ = logprob_stop
    ℓq = logpdf.(Ref(q), AdvancedVI.eachsample(samples_stop))
    f = ℓq - ℓπ
    return (mean(abs2, f) - mean(f)^2) / 2
end

function AdvancedVI.estimate_gradient!(
    rng::Random.AbstractRNG,
    obj::ScoreGradELBO,
    adtype::ADTypes.AbstractADType,
    out::DiffResults.MutableDiffResult,
    params,
    restructure,
    state,
    args...,
)
    q = restructure(params)
    (; obj_ad_prep, problem) = state
    samples = rand(rng, q, obj.n_samples)
    ℓπ = map(Base.Fix1(LogDensityProblems.logdensity, problem), eachsample(samples))
    aux = (adtype=adtype, logprob_stop=ℓπ, samples_stop=samples, restructure=restructure)
    AdvancedVI._value_and_gradient!(
        estimate_scoregradelbo_ad_forward, out, obj_ad_prep, adtype, params, aux
    )
    ℓq = logpdf.(Ref(q), AdvancedVI.eachsample(samples))
    elbo = mean(ℓπ - ℓq)
    stat = (elbo=elbo,)
    return out, state, stat
end
