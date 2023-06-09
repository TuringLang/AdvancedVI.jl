module AdvancedVI

using Random: Random

using Functors

using Distributions, DistributionsAD, Bijectors
using DocStringExtensions

using ProgressMeter, LinearAlgebra

using LogDensityProblems

using Distributions
using DistributionsAD

using StatsFuns
import StatsBase: entropy

using ForwardDiff, Tracker
import AbstractDifferentiation as AD

value_and_gradient(f, xs...; adbackend) = AD.value_and_gradient(adbackend, f, xs...)

const PROGRESS = Ref(true)
function turnprogress(switch::Bool)
    @info("[AdvancedVI]: global PROGRESS is set as $switch")
    PROGRESS[] = switch
end

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_ADVANCEDVI", "0")))

include("ad.jl")
include("utils.jl")

using Requires
function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        apply!(o, x, Δ) = Flux.Optimise.apply!(o, x, Δ)
        Flux.Optimise.apply!(o::TruncatedADAGrad, x, Δ) = apply!(o, x, Δ)
        Flux.Optimise.apply!(o::DecayedADAGrad, x, Δ) = apply!(o, x, Δ)
    end
end

export
    optimize,
    ELBO,
    ADVI,
    ADVIEnergy,
    ClosedFormEntropy,
    MonteCarloEntropy,
    LocationScale,
    FullRankGaussian,
    MeanFieldGaussian,
    TruncatedADAGrad,
    DecayedADAGrad

"""
    vi(model, alg::VariationalInference)
    vi(model, alg::VariationalInference, q::VariationalPosterior)
    vi(model, alg::VariationalInference, getq::Function, θ::AbstractArray)

Constructs the variational posterior from the `model` and performs the optimization
following the configuration of the given `VariationalInference` instance.

# Arguments
- `model`: `Turing.Model` or `Function` z ↦ log p(x, z) where `x` denotes the observations
- `alg`: the VI algorithm used
- `q`: a `VariationalPosterior` for which it is assumed a specialized implementation of the variational objective used exists.
- `getq`: function taking parameters `θ` as input and returns a `VariationalPosterior`
- `θ`: only required if `getq` is used, in which case it is the initial parameters for the variational posterior
"""
function vi end

function update end

# estimators
abstract type AbstractVariationalObjective end

include("objectives/elbo/elbo.jl")
include("objectives/elbo/advi_energy.jl")
include("objectives/elbo/entropy.jl")

# Variational Families
include("distributions/location_scale.jl")

# optimisers
include("optimisers.jl")

include("vi.jl")

end # module
