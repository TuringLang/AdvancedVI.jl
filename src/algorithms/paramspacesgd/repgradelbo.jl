
"""
    RepGradELBO(n_samples; kwargs...)

Evidence lower-bound objective with the reparameterization gradient formulation[^TL2014][^RMW2014][^KW2014].

# Arguments
- `n_samples::Int`: Number of Monte Carlo samples used to estimate the ELBO.

# Keyword Arguments
- `entropy`: The estimator for the entropy term. (Type `<: AbstractEntropyEstimator`; Default: `ClosedFormEntropy()`)

# Requirements
- The variational approximation ``q_{\\lambda}`` implements `rand`.
- The target distribution and the variational approximation have the same support.
- The target `LogDensityProblem` should satisfy either of the following: The target has a capability of at least `LogDensityProblems.LogDensityOrder{1}()` and the AD backend is one of `ReverseDiff`, `Zygote`, `Mooncake`, and `AutoEnzyme` in reverse mode so that `ADTypes.mode(adtype) == ADTypes.ReverseMode` is true. (In this case, `AdvancedVI` will take advantage of the existing `LogDensityProblems.logdensity_and_gradient`.) Otherwise, `LogDensityProblems.logdensity` should be differentiable under the selected AD backend.
- The sampling process `rand(q)` must be differentiable by the selected AD backend.

Depending on the options, additional requirements on ``q_{\\lambda}`` may apply.
"""
struct RepGradELBO{EntropyEst<:AbstractEntropyEstimator} <: AbstractVariationalObjective
    entropy::EntropyEst
    n_samples::Int
end

struct RepGradELBOState{Problem,ObjADPrep}
    problem::Problem
    obj_ad_prep::ObjADPrep
end

function init(
    rng::Random.AbstractRNG,
    obj::RepGradELBO,
    adtype::ADTypes.AbstractADType,
    prob::Prob,
    params,
    restructure,
) where {Prob}
    q_stop = restructure(params)
    capability = LogDensityProblems.capabilities(Prob)
    ad_prob = if capability < LogDensityProblems.LogDensityOrder{1}()
        @info "The capability of the supplied `LogDensityProblem` $(capability) is less than $(LogDensityProblems.LogDensityOrder{1}()). `AdvancedVI` will attempt to directly differentiate through `LogDensityProblems.logdensity`. If this is not intended, please supply a log-density problem with capability at least $(LogDensityProblems.LogDensityOrder{1}())"
        prob
    else
        if !(
            adtype isa Union{<:AutoReverseDiff,<:AutoZygote,<:AutoMooncake,<:AutoEnzyme} &&
            ADTypes.mode(adtype) isa ADTypes.ReverseMode
        )
            @info "The capability of the supplied target `LogDensityProblem` $(capability) is >= `LogDensityProblems.LogDensityOrder{1}()`. To make use of this, the `adtype` argument for AdvancedVI must be one of `AutoReverseDiff`, `AutoZygote`, `AutoMooncake`, or `AutoEnzyme` in reverse mode."
        end
        MixedADLogDensityProblem(prob)
    end
    aux = (
        rng=rng,
        adtype=adtype,
        obj=obj,
        problem=ad_prob,
        restructure=restructure,
        q_stop=q_stop,
    )
    obj_ad_prep = AdvancedVI._prepare_gradient(
        estimate_repgradelbo_ad_forward, adtype, params, aux
    )
    return RepGradELBOState(ad_prob, obj_ad_prep)
end

function RepGradELBO(n_samples::Int; entropy::AbstractEntropyEstimator=ClosedFormEntropy())
    return RepGradELBO(entropy, n_samples)
end

function Base.show(io::IO, obj::RepGradELBO)
    print(io, "RepGradELBO(entropy=")
    print(io, obj.entropy)
    print(io, ", n_samples=")
    print(io, obj.n_samples)
    return print(io, ")")
end

function estimate_energy_with_samples(prob, samples)
    return mean(Base.Fix1(LogDensityProblems.logdensity, prob), eachsample(samples))
end

"""
    reparam_with_entropy(rng, q, q_stop, n_samples, ent_est)

Draw `n_samples` from `q` and compute its entropy.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `q`: Variational approximation.
- `q_stop`: Same as `q`, but held constant during differentiation. Should only be used for computing the entropy.
- `n_samples::Int`: Number of Monte Carlo samples 
- `ent_est`: The entropy estimation strategy. (See `estimate_entropy`.)

# Returns
- `samples`: Monte Carlo samples generated through reparameterization. Their support matches that of the target distribution.
- `entropy`: An estimate (or exact value) of the differential entropy of `q`.
"""
function reparam_with_entropy(
    rng::Random.AbstractRNG, q, q_stop, n_samples::Int, ent_est::AbstractEntropyEstimator
)
    samples = rand(rng, q, n_samples)
    entropy = estimate_entropy(ent_est, samples, q, q_stop)
    return samples, entropy
end

function estimate_objective(
    rng::Random.AbstractRNG, obj::RepGradELBO, q, prob; n_samples::Int=obj.n_samples
)
    samples, entropy = reparam_with_entropy(rng, q, q, n_samples, obj.entropy)
    energy = estimate_energy_with_samples(prob, samples)
    return energy + entropy
end

function estimate_objective(obj::RepGradELBO, q, prob; n_samples::Int=obj.n_samples)
    return estimate_objective(Random.default_rng(), obj, q, prob; n_samples)
end

"""
    estimate_repgradelbo_ad_forward(params, aux)

AD-guaranteed forward path of the reparameterization gradient objective.

# Arguments
- `params`: Variational parameters.
- `aux`: Auxiliary information excluded from the AD path.

# Auxiliary Information 
`aux` should containt the following entries:
- `rng`: Random number generator.
- `obj`: The `RepGradELBO` objective.
- `problem`: The target `LogDensityProblem`.
- `adtype`: The `ADType` used for differentiating the forward path.
- `restructure`: Callable for restructuring the varitional distribution from `params`.
- `q_stop`: A copy of `restructure(params)` with its gradient "stopped" (excluded from the AD path).
"""
function estimate_repgradelbo_ad_forward(params, aux)
    (; rng, obj, problem, adtype, restructure, q_stop) = aux
    q = restructure_ad_forward(adtype, restructure, params)
    samples, entropy = reparam_with_entropy(rng, q, q_stop, obj.n_samples, obj.entropy)
    energy = estimate_energy_with_samples(problem, samples)
    elbo = energy + entropy
    return -elbo
end

function estimate_gradient!(
    rng::Random.AbstractRNG,
    obj::RepGradELBO,
    adtype::ADTypes.AbstractADType,
    out::DiffResults.MutableDiffResult,
    state::RepGradELBOState,
    params,
    restructure,
    args...,
)
    (; obj_ad_prep, problem) = state
    q_stop = restructure(params)
    aux = (
        rng=rng,
        adtype=adtype,
        obj=obj,
        problem=problem,
        restructure=restructure,
        q_stop=q_stop,
    )
    AdvancedVI._value_and_gradient!(
        estimate_repgradelbo_ad_forward, out, obj_ad_prep, adtype, params, aux
    )
    nelbo = DiffResults.value(out)
    info = (elbo=(-nelbo),)
    return out, state, info
end
