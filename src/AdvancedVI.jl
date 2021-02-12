module AdvancedVI

using Random: AbstractRNG

using Distributions, DistributionsAD, Bijectors
using DocStringExtensions

using ProgressMeter, LinearAlgebra

using ForwardDiff

const PROGRESS = Ref(true)
function turnprogress(switch::Bool)
    @info("[AdvancedVI]: global PROGRESS is set as $switch")
    PROGRESS[] = switch
end

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
        include(joinpath("compat", "zygote.jl"))
    end
    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        include(joinpath("compat", "reversediff.jl"))
    end
    @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
        include(joinpath("compat", "tracker.jl"))
    end
end

export
    vi,
    ADVI,
    ELBO,
    elbo,
    TruncatedADAGrad,
    DecayedADAGrad,
    VariationalInference

abstract type VariationalInference{AD} end

getchunksize(::Type{<:VariationalInference{AD}}) where AD = getchunksize(AD)
getADtype(::VariationalInference{AD}) where AD = AD

abstract type VariationalObjective end

const VariationalPosterior = Distribution{Multivariate, Continuous}


"""
    grad!(vo, alg::VariationalInference, q, model::Model, θ, out, args...)

Computes the gradients used in `optimize!`. Default implementation is provided for 
`VariationalInference{AD}` where `AD` is either `ForwardDiffAD` or `TrackerAD`.
This implicitly also gives a default implementation of `optimize!`.

Variance reduction techniques, e.g. control variates, should be implemented in this function.
"""
function grad! end

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


# Custom distributions
include("distributions.jl")

# objectives
include("gradients.jl")
include("interface.jl")
include("optimisers.jl")

# VI algorithms
include("dsvi.jl")

end # module
