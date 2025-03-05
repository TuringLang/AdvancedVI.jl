module AdvancedVI

using Accessors

using Random
using Distributions

using Functors
using Optimisers

using DocStringExtensions
using ProgressMeter
using LinearAlgebra

using LogDensityProblems

using ADTypes
using DiffResults
using DifferentiationInterface
using ChainRulesCore

using FillArrays

using StatsBase

# Derivatives
"""
    value_and_gradient!(ad, f, x, aux, out)

Evaluate the value and gradient of a function `f` at `x` using the automatic differentiation backend `ad` and store the result in `out`.
`f` may receive auxiliary input as `f(x,aux)`.

# Arguments
- `ad::ADTypes.AbstractADType`: Automatic differentiation backend. 
- `f`: Function subject to differentiation.
- `x`: The point to evaluate the gradient.
- `aux`: Auxiliary input passed to `f`.
- `out::DiffResults.MutableDiffResult`: Buffer to contain the output gradient and function value.
"""
function value_and_gradient!(
    ad::ADTypes.AbstractADType, f, x, aux, out::DiffResults.MutableDiffResult
)
    grad_buf = DiffResults.gradient(out)
    y, _ = DifferentiationInterface.value_and_gradient!(f, grad_buf, ad, x, Constant(aux))
    DiffResults.value!(out, y)
    return out
end

"""
    restructure_ad_forward(adtype, restructure, params)

Apply `restructure` to `params`.
This is an indirection for handling the type stability of `restructure`, as some AD backends require strict type stability in the AD path.

# Arguments
- `ad::ADTypes.AbstractADType`: Automatic differentiation backend. 
- `restructure`: Callable for restructuring the varitional distribution from `params`.
- `params`: Variational Parameters.
"""
restructure_ad_forward(::ADTypes.AbstractADType, restructure, params) = restructure(params)

# estimators
"""
    AbstractVariationalObjective

Abstract type for the VI algorithms supported by `AdvancedVI`.

# Implementations
To be supported by `AdvancedVI`, a VI algorithm must implement `AbstractVariationalObjective` and `estimate_objective`.
Also, it should provide gradients by implementing the function `estimate_gradient!`.
If the estimator is stateful, it can implement `init` to initialize the state.
"""
abstract type AbstractVariationalObjective end

"""
    init(rng, obj, prob, params, restructure)

Initialize a state of the variational objective `obj` given the initial variational parameters `λ`.
This function needs to be implemented only if `obj` is stateful.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `obj::AbstractVariationalObjective`: Variational objective.
- `params`: Initial variational parameters.
- `restructure`: Function that reconstructs the variational approximation from `λ`.
"""
init(::Random.AbstractRNG, ::AbstractVariationalObjective, ::Any, ::Any, ::Any) = nothing

"""
    estimate_objective([rng,] obj, q, prob; kwargs...)

Estimate the variational objective `obj` targeting `prob` with respect to the variational approximation `q`.

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
    estimate_gradient!(rng, obj, adtype, out, prob, params, restructure, obj_state)

Estimate (possibly stochastic) gradients of the variational objective `obj` targeting `prob` with respect to the variational parameters `λ`

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `obj::AbstractVariationalObjective`: Variational objective.
- `adtype::ADTypes.AbstractADType`: Automatic differentiation backend. 
- `out::DiffResults.MutableDiffResult`: Buffer containing the objective value and gradient estimates. 
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.
- `params`: Variational parameters to evaluate the gradient on.
- `restructure`: Function that reconstructs the variational approximation from `params`.
- `obj_state`: Previous state of the objective.

# Returns
- `out::MutableDiffResult`: Buffer containing the objective value and gradient estimates.
- `obj_state`: The updated state of the objective.
- `stat::NamedTuple`: Statistics and logs generated during estimation.
"""
function estimate_gradient! end

# ELBO-specific interfaces
abstract type AbstractEntropyEstimator end

"""
    estimate_entropy(entropy_estimator, mc_samples, q)

Estimate the entropy of `q`.

# Arguments
- `entropy_estimator`: Entropy estimation strategy.
- `q`: Variational approximation.
- `mc_samples`: Monte Carlo samples used to estimate the entropy. (Only used for Monte Carlo strategies.)

# Returns
- `obj_est`: Estimate of the objective value.
"""
function estimate_entropy end

export RepGradELBO,
    ScoreGradELBO, ClosedFormEntropy, StickingTheLandingEntropy, MonteCarloEntropy

include("objectives/elbo/entropy.jl")
include("objectives/elbo/repgradelbo.jl")
include("objectives/elbo/scoregradelbo.jl")

# Variational Families
export MvLocationScale, MeanFieldGaussian, FullRankGaussian

include("families/location_scale.jl")

export MvLocationScaleLowRank, LowRankGaussian

include("families/location_scale_low_rank.jl")

# Optimization Rules

include("optimization/rules.jl")

export DoWG, DoG, COCOB

# Output averaging strategy

abstract type AbstractAverager end

"""
    init(avg, params)

Initialize the state of the averaging strategy `avg` with the initial parameters `params`.

# Arguments
- `avg::AbstractAverager`: Averaging strategy.
- `params`: Initial variational parameters.
"""
init(::AbstractAverager, ::Any) = nothing

"""
    apply(avg::AbstractAverager, avg_st, params)

Apply averaging strategy `avg` on `params` given the state `avg_st`.

# Arguments
- `avg::AbstractAverager`: Averaging strategy.
- `avg_st`: Previous state of the averaging strategy.
- `params`: Initial variational parameters.
"""
function apply(::AbstractAverager, ::Any, ::Any) end

"""
    value(avg, avg_st)

Compute the output of the averaging strategy `avg` from the state `avg_st`.

# Arguments
- `avg::AbstractAverager`: Averaging strategy.
- `avg_st`: Previous state of the averaging strategy.
"""
function value(::AbstractAverager, ::Any) end

include("optimization/averaging.jl")

export NoAveraging, PolynomialAveraging

# Operators for Optimization
abstract type AbstractOperator end

"""
    apply(op::AbstractOperator, family, params, restructure)

Apply operator `op` on the variational parameters `params`. For instance, `op` could be a projection or proximal operator.

# Arguments
- `op::AbstractOperator`: Operator operating on the parameters `params`.
- `family::Type`: Type of the variational approximation `restructure(params)`.
- `params`: Variational parameters.
- `restructure`: Function that reconstructs the variational approximation from `params`.

# Returns
- `oped_params`: Parameters resulting from applying the operator.
"""
function apply(::AbstractOperator, ::Type, ::Any, ::Any) end

"""
    IdentityOperator()

Identity operator.
"""
struct IdentityOperator <: AbstractOperator end

apply(::IdentityOperator, ::Type, params, restructure) = params

include("optimization/clip_scale.jl")

export IdentityOperator, ClipScale

# Main optimization routine
"""
    optimize(problem, objective, q_init, max_iter, objargs...; kwargs...)              

Optimize the variational objective `objective` targeting the problem `problem` by estimating (stochastic) gradients.

The trainable parameters in the variational approximation are expected to be extractable through `Optimisers.destructure`.
This requires the variational approximation to be marked as a functor through `Functors.@functor`.

# Arguments
- `objective::AbstractVariationalObjective`: Variational Objective.
- `q_init`: Initial variational distribution. The variational parameters must be extractable through `Optimisers.destructure`.
- `max_iter::Int`: Maximum number of iterations.
- `objargs...`: Arguments to be passed to `objective`.

# Keyword Arguments
- `adtype::ADtypes.AbstractADType`: Automatic differentiation backend. 
- `optimizer::Optimisers.AbstractRule`: Optimizer used for inference. (Default: `Adam`.)
- `averager::AbstractAverager` : Parameter averaging strategy. (Default: `NoAveraging()`)
- `operator::AbstractOperator` : Operator applied to the parameters after each optimization step. (Default: `IdentityOperator()`)
- `rng::AbstractRNG`: Random number generator. (Default: `Random.default_rng()`.)
- `show_progress::Bool`: Whether to show the progress bar. (Default: `true`.)
- `callback`: Callback function called after every iteration. See further information below. (Default: `nothing`.)
- `prog`: Progress bar configuration. (Default: `ProgressMeter.Progress(n_max_iter; desc="Optimizing", barlen=31, showspeed=true, enabled=prog)`.)
- `state::NamedTuple`: Initial value for the internal state of optimization. Used to warm-start from the state of a previous run. (See the returned values below.)

# Returns
- `averaged_params`: Variational parameters generated by the algorithm averaged according to `averager`.
- `params`: Last variational parameters generated by the algorithm.
- `stats`: Statistics gathered during optimization.
- `state`: Collection of the final internal states of optimization. This can used later to warm-start from the last iteration of the corresponding run.

# Callback
The callback function `callback` has a signature of

    callback(; stat, state, params, averaged_params, restructure, gradient)

The arguments are as follows:
- `stat`: Statistics gathered during the current iteration. The content will vary depending on `objective`.
- `state`: Collection of the internal states used for optimization.
- `params`: Variational parameters.
- `averaged_params`: Variational parameters averaged according to the averaging strategy.
- `restructure`: Function that restructures the variational approximation from the variational parameters. Calling `restructure(param)` reconstructs the variational approximation. 
- `gradient`: The estimated (possibly stochastic) gradient.

`callback` can return a `NamedTuple` containing some additional information computed within `cb`.
This will be appended to the statistic of the current corresponding iteration.
Otherwise, just return `nothing`.

"""
function optimize(
    rng::Random.AbstractRNG,
    problem,
    objective::AbstractVariationalObjective,
    q_init,
    max_iter::Int,
    objargs...;
    adtype::ADTypes.AbstractADType,
    optimizer::Optimisers.AbstractRule=Optimisers.Adam(),
    averager::AbstractAverager=NoAveraging(),
    operator::AbstractOperator=IdentityOperator(),
    show_progress::Bool=true,
    state_init::NamedTuple=NamedTuple(),
    callback=nothing,
    prog=ProgressMeter.Progress(
        max_iter; desc="Optimizing", barlen=31, showspeed=true, enabled=show_progress
    ),
)
    params, restructure = Optimisers.destructure(deepcopy(q_init))
    opt_st = maybe_init_optimizer(state_init, optimizer, params)
    obj_st = maybe_init_objective(state_init, rng, objective, problem, params, restructure)
    avg_st = maybe_init_averager(state_init, averager, params)
    grad_buf = DiffResults.DiffResult(zero(eltype(params)), similar(params))
    stats = NamedTuple[]

    for t in 1:max_iter
        stat = (iteration=t,)
        grad_buf, obj_st, stat′ = estimate_gradient!(
            rng,
            objective,
            adtype,
            grad_buf,
            problem,
            params,
            restructure,
            obj_st,
            objargs...,
        )
        stat = merge(stat, stat′)

        grad = DiffResults.gradient(grad_buf)
        opt_st, params = Optimisers.update!(opt_st, params, grad)
        params = apply(operator, typeof(q_init), params, restructure)
        avg_st = apply(averager, avg_st, params)

        if !isnothing(callback)
            averaged_params = value(averager, avg_st)
            stat′ = callback(;
                stat,
                restructure,
                params=params,
                averaged_params=averaged_params,
                gradient=grad,
                state=(optimizer=opt_st, averager=avg_st, objective=obj_st),
            )
            stat = !isnothing(stat′) ? merge(stat′, stat) : stat
        end

        @debug "Iteration $t" stat...

        pm_next!(prog, stat)
        push!(stats, stat)
    end
    state = (optimizer=opt_st, averager=avg_st, objective=obj_st)
    stats = map(identity, stats)
    averaged_params = value(averager, avg_st)
    return restructure(averaged_params), restructure(params), stats, state
end

function optimize(
    problem,
    objective::AbstractVariationalObjective,
    q_init,
    max_iter::Int,
    objargs...;
    kwargs...,
)
    return optimize(
        Random.default_rng(), problem, objective, q_init, max_iter, objargs...; kwargs...
    )
end

export optimize

include("utils.jl")
include("optimize.jl")

# optional dependencies 
if !isdefined(Base, :get_extension) # check whether :get_extension is defined in Base
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Bijectors = "76274a88-744f-5084-9051-94815aaf08c4" begin
            include("../ext/AdvancedVIBijectorsExt.jl")
        end
        @require Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9" begin
            include("../ext/AdvancedVIEnzymeExt.jl")
        end
    end
end

end
