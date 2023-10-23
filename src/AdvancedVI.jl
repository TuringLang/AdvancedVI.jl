
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
    value_and_gradient!(
        ad::ADTypes.AbstractADType,
        f,
        θ::AbstractVector{<:Real},
        out::DiffResults.MutableDiffResult
    )

Evaluate the value and gradient of a function `f` at `θ` using the automatic differentiation backend `ad`.
The result is stored in `out`. 
The function `f` must return a scalar value.
The gradient is stored in `out` as a vector of the same length as `θ`.
"""
function value_and_gradient! end

# estimators
"""
    AbstractVariationalObjective

An VI algorithm supported by `AdvancedVI` should implement a subtype of  `AbstractVariationalObjective`.
Furthermore, it should implement the functions `estimate_gradient`.
If the estimator is stateful, it can implement `init` to initialize the state.
"""
abstract type AbstractVariationalObjective end

"""
    init(
        rng::Random.AbstractRNG,
        obj::AbstractVariationalObjective,
        λ::AbstractVector,
        restructure
    )

Initialize a state of the variational objective `obj` given the initial variational parameters `λ`.
This function needs to be implemented only if `obj` is stateful.
"""
init(
    ::Random.AbstractRNG,
    ::AbstractVariationalObjective,
    ::AbstractVector,
    ::Any
) = nothing

"""
    estimate_gradient!(
        rng         ::Random.AbstractRNG,
        obj         ::AbstractVariationalObjective,
        adbackend   ::ADTypes.AbstractADType,
        out         ::DiffResults.MutableDiffResult
        prob,
        λ,
        restructure,
        obj_state,
    )

Estimate (possibly stochastic) gradients of the objective `obj` targeting `prob` with respect to the variational parameters `λ` using the automatic differentiation backend `adbackend`.
The estimated objective value and gradient are then stored in `out`.
If the objective is stateful, `obj_state` is its previous state, otherwise, it is `nothing`.

# Returns
- `out`: The `MutableDiffResult` containing the objective value and gradient estimates.
- `obj_state`: The updated state of the objective estimator.
- `stat`: Statistics and logs generated during estimation. (Type: `<: NamedTuple`)
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

