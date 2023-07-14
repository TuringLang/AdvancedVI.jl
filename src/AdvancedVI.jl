
module AdvancedVI

using SimpleUnPack: @unpack, @pack!
using Accessors

import Random: AbstractRNG, default_rng
import Distributions: logpdf, _logpdf, rand, _rand!, _rand!

using Functors
using Optimisers

using DocStringExtensions

using ProgressMeter
using LinearAlgebra
using LinearAlgebra: AbstractTriangular

using LogDensityProblems

using ADTypes
using ADTypes: AbstractADType
using ForwardDiff, Tracker

using FillArrays
using PDMats
using Distributions, DistributionsAD
using Distributions: ContinuousMultivariateDistribution
using Bijectors

using StatsBase
using StatsBase: entropy

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_ADVANCEDVI", "0")))

using Requires
function __init__()
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        include("compat/zygote.jl")
    end
    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        include("compat/reversediff.jl")
    end
    @require Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9" begin
        include("compat/enzyme.jl")
    end
end

"""
    grad!(f, λ, out)

Computes the gradients of the objective f. Default implementation is provided for 
`VariationalInference{AD}` where `AD` is either `ForwardDiffAD` or `TrackerAD`.
This implicitly also gives a default implementation of `optimize!`.
"""
function grad! end

include("grad.jl")

# estimators
abstract type AbstractVariationalObjective end

function init              end
function estimate_gradient end

# ADVI-specific interfaces
abstract type AbstractEntropyEstimator end
abstract type AbstractControlVariate end

function update end
update(::Nothing, ::Nothing) = (nothing, nothing)

include("objectives/elbo/advi.jl")
include("objectives/elbo/entropy.jl")

export
    ELBO,
    ADVI,
    ADVIEnergy,
    ClosedFormEntropy,
    StickingTheLandingEntropy,
    MonteCarloEntropy

# Variational Families

include("distributions/location_scale.jl")

export
    VIFullRankGaussian,
    VIMeanFieldGaussian

"""
    optimize(model, alg::VariationalInference)
    optimize(model, alg::VariationalInference, q::VariationalPosterior)
    optimize(model, alg::VariationalInference, getq::Function, θ::AbstractArray)

Constructs the variational posterior from the `model` and performs the optimization
following the configuration of the given `VariationalInference` instance.

# Arguments
- `model`: `Turing.Model` or `Function` z ↦ log p(x, z) where `x` denotes the observations
- `alg`: the VI algorithm used
- `q`: a `VariationalPosterior` for which it is assumed a specialized implementation of the variational objective used exists.
- `getq`: function taking parameters `θ` as input and returns a `VariationalPosterior`
- `θ`: only required if `getq` is used, in which case it is the initial parameters for the variational posterior
"""
function optimize end

include("optimize.jl")

export optimize

include("utils.jl")

end # module
