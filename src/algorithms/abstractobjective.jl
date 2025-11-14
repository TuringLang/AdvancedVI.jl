
# estimators
"""
    AbstractVariationalObjective

Abstract type for a variational objective to be optimized by some variational algorithm.
"""
abstract type AbstractVariationalObjective end

"""
    init(rng, obj, adtype, q_init, prob, params, restructure)

Initialize a state of the variational objective `obj` given the initial variational approximation `q_init` and its parameters `params`.
This function needs to be implemented only if `obj` is stateful.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `obj::AbstractVariationalObjective`: Variational objective.
` `adtype::ADTypes.AbstractADType`: Automatic differentiation backend.
- `q_init`: Initial variational approximation.
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.
- `params`: Initial variational parameters.
- `restructure`: Function that reconstructs the variational approximation from `params`.
"""
function init(
    ::Random.AbstractRNG,
    ::AbstractVariationalObjective,
    ::ADTypes.AbstractADType,
    ::Any,
    ::Any,
    ::Any,
    ::Any,
)
    nothing
end

"""
    estimate_objective([rng,] obj, q, prob; kwargs...)

Estimate the minimization objective `obj` of the variational approximation `q` targeting `prob`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `obj::AbstractVariationalObjective`: Variational objective.
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.
- `q`: Variational approximation.

# Keyword Arguments
Depending on the objective, additional keyword arguments may apply.
Please refer to the respective documentation of each variational objective for more info.

# Returns
- `obj_est`: Estimate of the objective value.
"""
function estimate_objective end

export estimate_objective

"""
    set_objective_state_problem(state, prob)

Update the target problem object `prob` in the `state` of the associated objective.
This should be implemented for the objective to support `SubsampledObjective`.
"""
function set_objective_state_problem end

"""
    estimate_gradient!(rng, obj, adtype, out, obj_state, params, restructure)

Estimate (possibly stochastic) gradients of the variational objective `obj` with respect to the variational parameters `params`

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `obj::AbstractVariationalObjective`: Variational objective.
- `adtype::ADTypes.AbstractADType`: Automatic differentiation backend. 
- `out::DiffResults.MutableDiffResult`: Buffer containing the objective value and gradient estimates. 
- `obj_state`: Previous state of the objective.
- `params`: Variational parameters to evaluate the gradient on.
- `restructure`: Function that reconstructs the variational approximation from `params`.

# Returns
- `out::MutableDiffResult`: Buffer containing the objective value and gradient estimates.
- `obj_state`: The updated state of the objective.
- `stat::NamedTuple`: Statistics and logs generated during estimation.
"""
function estimate_gradient! end
