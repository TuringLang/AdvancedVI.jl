
module AdvancedVI

using SimpleUnPack: @unpack, @pack!
using Accessors

using Random: AbstractRNG, default_rng
using Distributions
import Distributions:
    logpdf, _logpdf, rand, rand!, _rand!,
    ContinuousMultivariateDistribution

using Functors
using Optimisers

using DocStringExtensions

using ProgressMeter
using LinearAlgebra
using LinearAlgebra: AbstractTriangular

using LogDensityProblems

using ADTypes, DiffResults
using ADTypes: AbstractADType
using ChainRulesCore: @ignore_derivatives 

using FillArrays
using Bijectors

using StatsBase
using StatsBase: entropy

# derivatives
"""
    value_and_gradient!(
        ad::ADTypes.AbstractADType,
        f,
        θ::AbstractVector{<:Real},
        out::DiffResults.MutableDiffResult
    )

Compute the value and gradient of a function `f` at `θ` using the automatic
differentiation backend `ad`.  The result is stored in `out`. 
The function `f` must return a scalar value. The gradient is stored in `out` as a
vector of the same length as `θ`.
"""
function value_and_gradient! end

# estimators
abstract type AbstractVariationalObjective end

function init              end
function estimate_gradient end

# ADVI-specific interfaces
abstract type AbstractEntropyEstimator end

# entropy.jl must preceed advi.jl
include("objectives/elbo/entropy.jl")
include("objectives/elbo/advi.jl")

export
    ELBO,
    ADVI,
    ClosedFormEntropy,
    StickingTheLandingEntropy,
    MonteCarloEntropy

# Variational Families

include("distributions/location_scale.jl")

export
    VILocationScale,
    VIFullRankGaussian,
    VIMeanFieldGaussian

# Optimization Routine

function optimize end

include("optimize.jl")

export optimize

include("utils.jl")


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

