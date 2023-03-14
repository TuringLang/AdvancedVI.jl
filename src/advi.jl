using StatsFuns
using DistributionsAD
using Bijectors
using Bijectors: TransformedDistribution


"""
$(TYPEDEF)

Automatic Differentiation Variational Inference (ADVI) with automatic differentiation
backend `AD`.

# Fields

$(TYPEDFIELDS)
"""
struct ADVI{AD} <: VariationalInference{AD}
    "Number of samples used to estimate the ELBO in each optimization step."
    samples_per_step::Int
    "Maximum number of gradient steps."
    max_iters::Int
end

function ADVI(samples_per_step::Int=1, max_iters::Int=1000)
    return ADVI{ADBackend()}(samples_per_step, max_iters)
end

alg_str(::ADVI) = "ADVI"

function vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
    θ = copy(θ_init)
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    # If `q` is a mean-field approx we use the specialized `update` function
    if q isa Distribution
        return update(q, θ)
    else
        # Otherwise we assume it's a mapping θ → q
        return q(θ)
    end
end


function optimize(elbo::ELBO, alg::ADVI, q, model, θ_init; optimizer = TruncatedADAGrad())
    θ = copy(θ_init)

    # `model` assumed to be callable z ↦ p(x, z)
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    return θ
end

