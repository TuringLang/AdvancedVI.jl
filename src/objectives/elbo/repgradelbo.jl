
"""
    RepGradELBO(n_samples; kwargs...)

Evidence lower-bound objective with the reparameterization gradient formulation[^TL2014][^RMW2014][^KW2014].
This computes the evidence lower-bound (ELBO) through the formulation:
```math
\\begin{aligned}
\\mathrm{ELBO}\\left(\\lambda\\right)
&\\triangleq
\\mathbb{E}_{z \\sim q_{\\lambda}}\\left[
  \\log \\pi\\left(z\\right)
\\right]
+ \\mathbb{H}\\left(q_{\\lambda}\\right),
\\end{aligned}
```

# Arguments
- `n_samples::Int`: Number of Monte Carlo samples used to estimate the ELBO.

# Keyword Arguments
- `entropy`: The estimator for the entropy term. (Type `<: AbstractEntropyEstimator`; Default: `ClosedFormEntropy()`)

# Requirements
- The variational approximation ``q_{\\lambda}`` implements `rand`.
- The target distribution and the variational approximation have the same support.
- The target `logdensity(prob, x)` must be differentiable with respect to `x` by the selected AD backend.

Depending on the options, additional requirements on ``q_{\\lambda}`` may apply.
"""
struct RepGradELBO{EntropyEst<:AbstractEntropyEstimator} <: AbstractVariationalObjective
    entropy::EntropyEst
    n_samples::Int
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

function estimate_entropy_maybe_stl(
    entropy_estimator::AbstractEntropyEstimator, samples, q, q_stop
)
    q_maybe_stop = maybe_stop_entropy_score(entropy_estimator, q, q_stop)
    return estimate_entropy(entropy_estimator, samples, q_maybe_stop)
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
    entropy = estimate_entropy_maybe_stl(ent_est, samples, q, q_stop)
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

function estimate_repgradelbo_ad_forward(params′, aux)
    @unpack rng, obj, problem, adtype, restructure, q_stop = aux
    q = restructure_ad_forward(adtype, restructure, params′)
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
    prob,
    params,
    restructure,
    state,
)
    q_stop = restructure(params)
    aux = (
        rng=rng,
        adtype=adtype,
        obj=obj,
        problem=prob,
        restructure=restructure,
        q_stop=q_stop,
    )
    value_and_gradient!(adtype, estimate_repgradelbo_ad_forward, params, aux, out)
    nelbo = DiffResults.value(out)
    stat = (elbo=-nelbo,)
    return out, nothing, stat
end
