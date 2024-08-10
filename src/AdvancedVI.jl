
module AdvancedVI

using SimpleUnPack: @unpack, @pack!
using Accessors

using Random
using Distributions

using Functors
using Optimisers

using DocStringExtensions
using ProgressMeter
using LinearAlgebra

using LogDensityProblems

using ADTypes, DiffResults
using ChainRulesCore

using FillArrays

using StatsBase

# derivatives
"""
    value_and_gradient!(ad, f, x, out)
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
function value_and_gradient! end

"""
    stop_gradient(x)

Stop the gradient from propagating to `x` if the selected ad backend supports it.
Otherwise, it is equivalent to `identity`.

# Arguments
- `x`: Input

# Returns
- `x`: Same value as the input.
"""
function stop_gradient end

# Update for gradient descent step
"""
    update_variational_params!(family_type, opt_st, params, restructure, grad)

Update variational distribution according to the update rule in the optimizer state `opt_st` and the variational family `family_type`.

This is a wrapper around `Optimisers.update!` to provide some indirection.
For example, depending on the optimizer and the variational family, this may do additional things such as applying projection or proximal mappings.
Same as the default behavior of `Optimisers.update!`, `params` and `opt_st` may be updated by the routine and are no longer valid after calling this functino.
Instead, the return values should be used.

# Arguments
- `family_type::Type`: Type of the variational family `typeof(restructure(params))`.
- `opt_st`: Optimizer state returned by `Optimisers.setup`.
- `params`: Current set of parameters to be updated.
- `restructure`: Callable for restructuring the varitional distribution from `params`.
- `grad`: Gradient to be used by the update rule of `opt_st`.

# Returns
- `opt_st`: Updated optimizer state.
- `params`: Updated parameters.
"""
function update_variational_params! end

function update_variational_params!(::Type, opt_st, params, restructure, grad)
    return Optimisers.update!(opt_st, params, grad)
end

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
    estimate_gradient!(rng, obj, adtype, out, prob, λ, restructure, obj_state)

Estimate (possibly stochastic) gradients of the variational objective `obj` targeting `prob` with respect to the variational parameters `λ`

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `obj::AbstractVariationalObjective`: Variational objective.
- `adtype::ADTypes.AbstractADType`: Automatic differentiation backend. 
- `out::DiffResults.MutableDiffResult`: Buffer containing the objective value and gradient estimates. 
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.
- `λ`: Variational parameters to evaluate the gradient on.
- `restructure`: Function that reconstructs the variational approximation from `λ`.
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

export RepGradELBO, ClosedFormEntropy, StickingTheLandingEntropy, MonteCarloEntropy

include("objectives/elbo/entropy.jl")
include("objectives/elbo/repgradelbo.jl")

# Variational Families
export MvLocationScale, MeanFieldGaussian, FullRankGaussian

include("families/location_scale.jl")

export MvLocationScaleLowRank, LowRankGaussian

include("families/location_scale_low_rank.jl")

# Optimization Routine

function optimize end

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
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
            include("../ext/AdvancedVIForwardDiffExt.jl")
        end
        @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
            include("../ext/AdvancedVIReverseDiffExt.jl")
        end
        @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
            include("../ext/AdvancedVIZygoteExt.jl")
        end
    end
end

end
