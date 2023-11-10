
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
using Bijectors

using StatsBase

# derivatives
"""
    value_and_gradient!(ad, f, θ, out)

Evaluate the value and gradient of a function `f` at `θ` using the automatic differentiation backend `ad` and store the result in `out`.

# Arguments
- `ad::ADTypes.AbstractADType`: Automatic differentiation backend. 
- `f`: Function subject to differentiation.
- `θ`: The point to evaluate the gradient.
- `out::DiffResults.MutableDiffResult`: Buffer to contain the output gradient and function value.
"""
function value_and_gradient! end

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
    init(rng, obj, λ, restructure)

Initialize a state of the variational objective `obj` given the initial variational parameters `λ`.
This function needs to be implemented only if `obj` is stateful.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `obj::AbstractVariationalObjective`: Variational objective.
- `λ`: Initial variational parameters.
- `restructure`: Function that reconstructs the variational approximation from `λ`.
"""
init(
    ::Random.AbstractRNG,
    ::AbstractVariationalObjective,
    ::AbstractVector,
    ::Any
) = nothing

"""
    estimate_objective([rng,] obj, q, prob, kwargs...)

Estimate the variational objective `obj` targeting `prob` with respect to the variational approximation `q`.

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `obj::AbstractVariationalObjective`: Variational objective.
- `prob`: The target log-joint likelihood implementing the `LogDensityProblem` interface.
- `q`: Variational approximation.

# Keyword Arguments
For the keywword arguments, refer to the respective documentation for each variational objective.

# Returns
- `obj_est`: Estimate of the objective value.
"""
function estimate_objective end

export estimate_objective


"""
    estimate_gradient!(rng, obj, adbackend, out, prob, λ, restructure, obj_state)

Estimate (possibly stochastic) gradients of the variational objective `obj` targeting `prob` with respect to the variational parameters `λ`

# Arguments
- `rng::Random.AbstractRNG`: Random number generator.
- `obj::AbstractVariationalObjective`: Variational objective.
- `adbackend::ADTypes.AbstractADType`: Automatic differentiation backend. 
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

# ADVI-specific interfaces
abstract type AbstractEntropyEstimator end

export
    ADVI,
    ClosedFormEntropy,
    StickingTheLandingEntropy,
    MonteCarloEntropy

# entropy.jl must preceed advi.jl
include("objectives/elbo/entropy.jl")
include("objectives/elbo/advi.jl")

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

