module AdvancedVI

using Random: Random

using Functors

using Optimisers

using DocStringExtensions

using ProgressMeter
using LinearAlgebra
using LinearAlgebra: AbstractTriangular

using LogDensityProblems

using ForwardDiff, Tracker

using FillArrays
using PDMats
using Distributions, DistributionsAD
using Distributions: ContinuousMultivariateDistribution
using Bijectors

using StatsBase
using StatsBase: entropy

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_ADVANCEDVI", "0")))

include("ad.jl")

using Requires
function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        apply!(o, x, Δ) = Flux.Optimise.apply!(o, x, Δ)
        Flux.Optimise.apply!(o::TruncatedADAGrad, x, Δ) = apply!(o, x, Δ)
        Flux.Optimise.apply!(o::DecayedADAGrad, x, Δ) = apply!(o, x, Δ)
    end
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        include("compat/zygote.jl")
        export ZygoteAD

        function AdvancedVI.grad!(
            f::Function,
            ::Type{<:ZygoteAD},
            λ::AbstractVector{<:Real},
            out::DiffResults.MutableDiffResult,
        )
            y, back = Zygote.pullback(f, λ)
            dy = first(back(1.0))
            DiffResults.value!(out, y)
            DiffResults.gradient!(out, dy)
            return out
        end
    end
    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        include("compat/reversediff.jl")
        export ReverseDiffAD

        function AdvancedVI.grad!(
            f::Function,
            ::Type{<:ReverseDiffAD},
            λ::AbstractVector{<:Real},
            out::DiffResults.MutableDiffResult,
        )
            tp = AdvancedVI.tape(f, λ)
            ReverseDiff.gradient!(out, tp, λ)
            return out
        end
    end
    @require Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9" begin
        include("compat/enzyme.jl")
        export EnzymeAD

        function AdvancedVI.grad!(
            f::Function,
            ::Type{<:EnzymeAD},
            λ::AbstractVector{<:Real},
            out::DiffResults.MutableDiffResult,
        )
            # Use `Enzyme.ReverseWithPrimal` once it is released:
            # https://github.com/EnzymeAD/Enzyme.jl/pull/598
            y = f(λ)
            DiffResults.value!(out, y)
            dy = DiffResults.gradient(out)
            fill!(dy, 0)
            Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(λ, dy))
            return out
        end
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
    grad!(f, λ, out)

Computes the gradients of the objective f. Default implementation is provided for 
`VariationalInference{AD}` where `AD` is either `ForwardDiffAD` or `TrackerAD`.
This implicitly also gives a default implementation of `optimize!`.
"""
function grad! end

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

function update end

# default implementations
function grad!(
    f::Function,
    adtype::Type{<:ForwardDiffAD},
    λ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult
)
    # Set chunk size and do ForwardMode.
    chunk_size = getchunksize(adtype)
    config = if chunk_size == 0
        ForwardDiff.GradientConfig(f, λ)
    else
        ForwardDiff.GradientConfig(f, λ, ForwardDiff.Chunk(length(λ), chunk_size))
    end
    ForwardDiff.gradient!(out, f, λ, config)
end

function grad!(
    f::Function,
    ::Type{<:TrackerAD},
    λ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult
)
    λ_tracked = Tracker.param(λ)
    y = f(λ_tracked)
    Tracker.back!(y, 1.0)

    DiffResults.value!(out, Tracker.data(y))
    DiffResults.gradient!(out, Tracker.grad(λ_tracked))
end

# estimators
abstract type AbstractVariationalObjective end

function estimate_gradient end

abstract type AbstractEnergyEstimator  end
abstract type AbstractEntropyEstimator end
abstract type AbstractControlVariate end

init(::Nothing) = nothing

update(::Nothing, ::Nothing) = (nothing, nothing)

include("objectives/elbo/advi.jl")
include("objectives/elbo/advi_energy.jl")
include("objectives/elbo/entropy.jl")

# Variational Families
include("distributions/location_scale.jl")

# optimisers
include("optimisers.jl")

include("utils.jl")
include("vi.jl")

end # module
