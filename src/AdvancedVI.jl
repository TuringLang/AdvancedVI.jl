module AdvancedVI

using Random: AbstractRNG

using Distributions, DistributionsAD, Bijectors

# TODO: remove in favour of fix in DistributionsAD when that's done
Base.size(d::TuringDiagMvNormal) = (length(d), )

using ProgressMeter, LinearAlgebra

using ForwardDiff
using Tracker

const PROGRESS = Ref(true)
function turnprogress(switch::Bool)
    @info("[Turing]: global PROGRESS is set as $switch")
    PROGRESS[] = switch
end

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_ADVANCEDVI", "0")))

include("ad.jl")

using Requires
function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" apply!(o, x, Δ) = Flux.Optimise.apply!(o, x, Δ)
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" Flux.Optimise.apply!(o::TruncatedADAGrad, x, Δ) = apply!(o, x, Δ)
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" Flux.Optimise.apply!(o::DecayedADAGrad, x, Δ) = apply!(o, x, Δ)
end

export
    vi,
    ADVI,
    ELBO,
    elbo,
    TruncatedADAGrad,
    DecayedADAGrad

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
function grad!(
    vo,
    alg::VariationalInference,
    q,
    model,
    θ,
    out,
    args...
)
    error("Turing.Variational.grad!: unmanaged variational inference algorithm: "
          * "$(typeof(alg))")
end

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
function vi(model, alg::VariationalInference)
    error("Turing.Variational.vi: variational inference algorithm $(typeof(alg)) "
          * "is not implemented")
end
function vi(model, alg::VariationalInference, q)
    error("Turing.Variational.vi: variational inference algorithm $(typeof(alg)) "
          * "is not implemented")
end
function vi(model, alg::VariationalInference, q, θ_init)
    error("Turing.Variational.vi: variational inference algorithm $(typeof(alg)) "
          * "is not implemented")
end

# default implementations
function grad!(
    vo,
    alg::VariationalInference{<:ForwardDiffAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    f(θ_) = if (q isa VariationalPosterior)
        - vo(alg, update(q, θ_), model, args...)
    else
        - vo(alg, q(θ_), model, args...)
    end

    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
    config = ForwardDiff.GradientConfig(f, θ, chunk)
    ForwardDiff.gradient!(out, f, θ, config)
end

function grad!(
    vo,
    alg::VariationalInference{<:TrackerAD},
    q,
    model,
    θ::AbstractVector{<:Real},
    out::DiffResults.MutableDiffResult,
    args...
)
    θ_tracked = Tracker.param(θ)
    y = if (q isa VariationalPosterior)
        - vo(alg, update(q, θ_tracked), model, args...)
    else
        - vo(alg, q(θ_tracked), model, args...)
    end
    Tracker.back!(y, 1.0)

    DiffResults.value!(out, Tracker.data(y))
    DiffResults.gradient!(out, Tracker.grad(θ_tracked))
end

import Tracker: TrackedArray, track, Call
function TrackedArray(f::Call, x::SA) where {T, N, A, SA<:SubArray{T, N, A}}
    TrackedArray(f, convert(A, x))
end

"""
    optimize!(vo, alg::VariationalInference{AD}, q::VariationalPosterior, model::Model, θ; optimizer = TruncatedADAGrad())

Iteratively updates parameters by calling `grad!` and using the given `optimizer` to compute
the steps.
"""
function optimize!(
    vo,
    alg::VariationalInference,
    q,
    model,
    θ::AbstractVector{<:Real};
    optimizer = TruncatedADAGrad()
)
    # TODO: should we always assume `samples_per_step` and `max_iters` for all algos?
    alg_name = alg_str(alg)
    samples_per_step = alg.samples_per_step
    max_iters = alg.max_iters
    
    num_params = length(θ)

    # TODO: really need a better way to warn the user about potentially
    # not using the correct accumulator
    if (optimizer isa TruncatedADAGrad) && (θ ∉ keys(optimizer.acc))
        # this message should only occurr once in the optimization process
        @info "[$alg_name] Should only be seen once: optimizer created for θ" objectid(θ)
    end

    diff_result = DiffResults.GradientResult(θ)

    i = 0
    prog = if PROGRESS[]
        ProgressMeter.Progress(max_iters, 1, "[$alg_name] Optimizing...", 0)
    else
        0
    end

    # add criterion? A running mean maybe?
    time_elapsed = @elapsed while (i < max_iters) # & converged
        grad!(vo, alg, q, model, θ, diff_result, samples_per_step)

        # apply update rule
        Δ = DiffResults.gradient(diff_result)
        Δ = apply!(optimizer, θ, Δ)
        @. θ = θ - Δ
        
        AdvancedVI.DEBUG && @debug "Step $i" Δ DiffResults.value(diff_result)
        PROGRESS[] && (ProgressMeter.next!(prog))

        i += 1
    end

    return θ
end

# utilities
update(d::TuringDiagMvNormal, μ, σ) = TuringDiagMvNormal(μ, σ)
update(td::TransformedDistribution, θ...) = transformed(update(td.dist, θ...), td.transform)
function update(td::TransformedDistribution{<:TuringDiagMvNormal}, θ::AbstractArray)
    μ, ω = θ[1:length(td)], θ[length(td) + 1:end]
    return update(td, μ, softplus.(ω))
end

# TODO: add these to DistributionsAD.jl and remove from here
Distributions.params(d::TuringDiagMvNormal) = (d.m, d.σ)

import StatsBase: entropy
function entropy(d::TuringDiagMvNormal)
    T = eltype(d.σ)
    return (DistributionsAD.length(d) * (T(log2π) + one(T)) / 2 + sum(log.(d.σ)))
end


# objectives
include("objectives.jl")

# optimisers
include("optimisers.jl")

# VI algorithms
include("advi.jl")

end # module
