
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

using FillArrays

using StatsBase

# Derivatives
"""
    _value_and_gradient!(f, out, ad, x, aux)
    _value_and_gradient!(f, out, prep, ad, x, aux)

Evaluate the value and gradient of a function `f` at `x` using the automatic differentiation backend `ad` and store the result in `out`.
`f` may receive auxiliary input as `f(x,aux)`.

# Arguments
- `ad::ADTypes.AbstractADType`: 
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
- `prep`: Output of `DifferentiationInterface.prepare_gradient`.
- `out::DiffResults.MutableDiffResult`: Buffer to contain the output gradient and function value.
"""
function _value_and_gradient!(
    f, out::DiffResults.MutableDiffResult, ad::ADTypes.AbstractADType, x, aux
)
    grad_buf = DiffResults.gradient(out)
    y, _ = DifferentiationInterface.value_and_gradient!(f, grad_buf, ad, x, Constant(aux))
    DiffResults.value!(out, y)
    return out
end

function _value_and_gradient!(
    f, out::DiffResults.MutableDiffResult, prep, ad::ADTypes.AbstractADType, x, aux
)
    grad_buf = DiffResults.gradient(out)
    y, _ = DifferentiationInterface.value_and_gradient!(
        f, grad_buf, prep, ad, x, Constant(aux)
    )
    DiffResults.value!(out, y)
    return out
end

"""
    _prepare_gradient!(f, ad, x, aux)

Prepare AD backend for taking gradients of a function `f` at `x` using the automatic differentiation backend `ad`.

# Arguments
- `ad::ADTypes.AbstractADType`: Automatic differentiation backend.
- `f`: Function subject to differentiation.
- `x`: The point to evaluate the gradient.
- `aux`: Auxiliary input passed to `f`.
"""
function _prepare_gradient(f, ad::ADTypes.AbstractADType, x, aux)
    return DifferentiationInterface.prepare_gradient(f, ad, x, Constant(aux))
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
    apply(op::AbstractOperator, family, rule, opt_state, params, restructure)

Apply operator `op` on the variational parameters `params`. For instance, `op` could be a projection or proximal operator.

# Arguments
- `op::AbstractOperator`: Operator operating on the parameters `params`.
- `family::Type`: Type of the variational approximation `restructure(params)`.
- `opt_state`: State of the optimizer.
- `params`: Variational parameters.
- `restructure`: Function that reconstructs the variational approximation from `params`.

# Returns
- `oped_params`: Parameters resulting from applying the operator.
"""
function apply(::AbstractOperator, ::Type, ::Optimisers.AbstractRule, ::Any, ::Any, ::Any) end

"""
    IdentityOperator()

Identity operator.
"""
struct IdentityOperator <: AbstractOperator end

apply(::IdentityOperator, ::Type, opt_st, params, restructure) = params

include("optimization/clip_scale.jl")
include("optimization/proximal_location_scale_entropy.jl")

export IdentityOperator, ClipScale, ProximalLocationScaleEntropy

# Algorithms

"""
    AbstractAlgorithm

Abstract type for a variational inference algorithm.
"""
abstract type AbstractAlgorithm end

"""
    init(rng, alg, prob, q_init)

Initialize `alg` given the initial variational approximation `q_init` and the target `prob`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `alg::AbstractAlgorithm`: Variational inference algorithm.
- `prob`: Target problem.
` `q_init`: Initial variational approximation.
"""
init(::Random.AbstractRNG, ::AbstractAlgorithm, ::Any, ::Any) = nothing

"""
    step(rng, alg, state, callback, objargs...; kwargs...)

Perform a single step of `alg` given the previous `stat`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `alg::AbstractAlgorithm`: Variational inference algorithm.
- `state`: Previous state of the algorithm.
- `callback`: Callback function to be called during the step.

# Returns
- state: New state generated by performing the step.
- terminate::Bool: Whether to terminate the algorithm after the step.
- info::NamedTuple: Information generated during the step. 
"""
function step(
    ::Random.AbstractRNG, ::AbstractAlgorithm, ::Any, callback, objargs...; kwargs...
)
    nothing
end

"""
    output(alg, state)

Generate an output variational approximation using the last `state` of `alg`.

# Arguments
- `alg::AbstractAlgorithm`: Variational inference algorithm used to compute the state.
- `state`: The last state generated by the algorithm.

# Returns
- `out`: The output of the algorithm. 
"""
output(::AbstractAlgorithm, ::Any) = nothing

# Main optimization routine
function optimize end

export optimize

include("utils.jl")
include("optimize.jl")

## Parameter Space SGD
include("algorithms/paramspacesgd/abstractobjective.jl")
include("algorithms/paramspacesgd/paramspacesgd.jl")

export ParamSpaceSGD

## Parameter Space SGD Implementations
### ELBO Maximization

abstract type AbstractEntropyEstimator end

"""
    estimate_entropy(entropy_estimator, mc_samples, q, q_stop)

Estimate the entropy of `q`.

# Arguments
- `entropy_estimator`: Entropy estimation strategy.
- `q`: Variational approximation.
- `q_stop`: Variational approximation with detached from the automatic differentiation graph.
- `mc_samples`: Monte Carlo samples used to estimate the entropy. (Only used for Monte Carlo strategies.)

# Returns
- `obj_est`: Estimate of the objective value.
"""
function estimate_entropy end

include("algorithms/paramspacesgd/repgradelbo.jl")
include("algorithms/paramspacesgd/scoregradelbo.jl")
include("algorithms/paramspacesgd/entropy.jl")

export RepGradELBO,
    ScoreGradELBO,
    ClosedFormEntropy,
    StickingTheLandingEntropy,
    MonteCarloEntropy,
    ClosedFormEntropyZeroGradient,
    StickingTheLandingEntropyZeroGradient

include("algorithms/paramspacesgd/constructors.jl")

export BBVIRepGrad, BBVIRepGradProxLocScale

end
