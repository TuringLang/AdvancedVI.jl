
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
- `ADbackend::ADTypes.AbstractADType=ADTypes.AutoZygote()`: 
    automatic differentiation backend. Currently supports
    `ADTypes.AutoZygote()`, `ADTypes.ForwardDiff()`, `ADTypes.ReverseDiff()`, 
    `ADTypes.AutoMooncake()` and
    `ADTypes.AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Const,
    )`.
    If one wants to use `AutoEnzyme`, please make sure to include the `set_runtime_activity` and `function_annotation` as shown above.
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
function optimize end

export optimize

include("utils.jl")
include("optimize.jl")

end
