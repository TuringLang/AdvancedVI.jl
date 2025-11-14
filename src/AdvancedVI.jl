
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
using ChainRulesCore: ChainRulesCore

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
- `restructure`: Callable for restructuring the variational distribution from `params`.
- `params`: Variational Parameters.
"""
restructure_ad_forward(::ADTypes.AbstractADType, restructure, params) = restructure(params)

include("mixedad_logdensity.jl")

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
    AbstractVariationalAlgorithm

Abstract type for a variational inference algorithm.
"""
abstract type AbstractVariationalAlgorithm end

"""
    init(rng, alg, q_init, prob)

Initialize `alg` given the initial variational approximation `q_init` and the target `prob`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `alg::AbstractVariationalAlgorithm`: Variational inference algorithm.
- `q_init`: Initial variational approximation.
- `prob`: Target problem.

# Returns
- `state`: Initial state of the algorithm.
"""
init(::Random.AbstractRNG, ::AbstractVariationalAlgorithm, ::Any, ::Any) = nothing

"""
    step(rng, alg, state, callback, objargs...; kwargs...)

Perform a single step of `alg` given the previous `state`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `alg::AbstractVariationalAlgorithm`: Variational inference algorithm.
- `state`: Previous state of the algorithm.
- `callback`: Callback function to be called during the step.

# Returns
- `state`: New state generated by performing the step.
- `terminate`::Bool: Whether to terminate the algorithm after the step.
- `info`::NamedTuple: Information generated during the step. 
"""
function step(
    ::Random.AbstractRNG,
    ::AbstractVariationalAlgorithm,
    ::Any,
    callback,
    objargs...;
    kwargs...,
)
    nothing
end

"""
    output(alg, state)

Output a variational approximation from the last `state` of `alg`.

# Arguments
- `alg::AbstractVariationalAlgorithm`: Variational inference algorithm used to compute the state.
- `state`: The last state generated by the algorithm.

# Returns
- `out`: The output of the algorithm. 
"""
output(::AbstractVariationalAlgorithm, ::Any) = nothing

"""
    estimate_objective([rng,] alg, q, prob; kwargs...)

Estimate the variational objective subject to be minimized by the algorithm `alg` for approximating the target `prob` with the variational approximation `q`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `alg::AbstractVariationalAlgorithm`: Variational inference algorithm.
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.
- `q`: Variational approximation.

# Keyword Arguments
Depending on the algorithm, additional keyword arguments may apply.
Please refer to the respective documentation of each algorithm for more info.

# Returns
- `obj_est`: Estimate of the objective value.
"""
function estimate_objective(
    ::Random.AbstractRNG, ::AbstractVariationalAlgorithm, q, prob; kwargs...
)
    nothing
end

function estimate_objective(alg::AbstractVariationalAlgorithm, q, prob; kwargs...)
    estimate_objective(Random.default_rng(), alg, q, prob; kwargs...)
end

export estimate_objective

# Subsampling
"""
    subsample(model, batch)
    subsample(q, batch)

Inform `model` or `q` to only use the data points designated by the iterable collection `batch`.
For `model`, the log-density should also be adjusted to account for the change in number of data points.
"""
subsample(model_or_q::Any, ::Any) = model_or_q

abstract type AbstractSubsampling end

include("reshuffling.jl")

export ReshufflingBatchSubsampling

# Main optimization routine
function optimize end

export optimize

include("utils.jl")
include("optimize.jl")

## Parameter Space SGD Implementations

include("algorithms/abstractobjective.jl")

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

include("algorithms/subsampledobjective.jl")
include("algorithms/repgradelbo.jl")
include("algorithms/scoregradelbo.jl")
include("algorithms/entropy.jl")

export RepGradELBO,
    ScoreGradELBO,
    ClosedFormEntropy,
    StickingTheLandingEntropy,
    MonteCarloEntropy,
    ClosedFormEntropyZeroGradient,
    StickingTheLandingEntropyZeroGradient,
    SubsampledObjective

include("algorithms/constructors.jl")
include("algorithms/common.jl")

export KLMinRepGradDescent, KLMinRepGradProxDescent, KLMinScoreGradDescent, ADVI, BBVI

# Natural and Wasserstein gradient descent algorithms

include("algorithms/gauss_expected_grad_hess.jl")
include("algorithms/klminwassfwdbwd.jl")
include("algorithms/klminsqrtnaturalgraddescent.jl")
include("algorithms/klminnaturalgraddescent.jl")

export KLMinWassFwdBwd, KLMinSqrtNaturalGradDescent, KLMinNaturalGradDescent

end
