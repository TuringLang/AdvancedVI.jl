using StatsFuns
using DistributionsAD
using Bijectors
using Bijectors: TransformedDistribution
using Random: AbstractRNG, GLOBAL_RNG

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


"""
    ADVI(samples_per_step = 1, max_iters = 1000)

Automatic Differentiation Variational Inference (ADVI) for a given model.
"""
struct ADVI{AD} <: VariationalInference{AD}
    samples_per_step # number of samples used to estimate the ELBO in each optimization step
    max_iters        # maximum number of gradient steps used in optimization
end

ADVI(args...) = ADVI{ADBackend()}(args...)
ADVI() = ADVI(1, 1000)

alg_str(::ADVI) = "ADVI"


function vi(model, alg::ADVI, q::TransformedDistribution{<:TuringDiagMvNormal}; optimizer = TruncatedADAGrad())
    DEBUG && @debug "Optimizing ADVI..."
    # Initial parameters for mean-field approx
    μ, σs = params(q)
    θ = vcat(μ, invsoftplus.(σs))

    # Optimize
    optimize!(elbo, alg, q, model, θ; optimizer = optimizer)

    # Return updated `Distribution`
    return update(q, θ)
end

function vi(model, alg::ADVI, q, θ_init; optimizer = TruncatedADAGrad())
    DEBUG && @debug "Optimizing ADVI..."
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

# WITHOUT updating parameters inside ELBO
function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::ADVI,
    q::VariationalPosterior,
    logπ,
    num_samples
)
    #   𝔼_q(z)[log p(xᵢ, z)]
    # = ∫ log p(xᵢ, z) q(z) dz
    # = ∫ log p(xᵢ, f(ϕ)) q(f(ϕ)) |det J_f(ϕ)| dϕ   (since change of variables)
    # = ∫ log p(xᵢ, f(ϕ)) q̃(ϕ) dϕ                   (since q(f(ϕ)) |det J_f(ϕ)| = q̃(ϕ))
    # = 𝔼_q̃(ϕ)[log p(xᵢ, z)]

    #   𝔼_q(z)[log q(z)]
    # = ∫ q(f(ϕ)) log (q(f(ϕ))) |det J_f(ϕ)| dϕ     (since q(f(ϕ)) |det J_f(ϕ)| = q̃(ϕ))
    # = 𝔼_q̃(ϕ) [log q(f(ϕ))]
    # = 𝔼_q̃(ϕ) [log q̃(ϕ) - log |det J_f(ϕ)|]
    # = 𝔼_q̃(ϕ) [log q̃(ϕ)] - 𝔼_q̃(ϕ) [log |det J_f(ϕ)|]
    # = - ℍ(q̃(ϕ)) - 𝔼_q̃(ϕ) [log |det J_f(ϕ)|]

    # Finally, the ELBO is given by
    # ELBO = 𝔼_q(z)[log p(xᵢ, z)] - 𝔼_q(z)[log q(z)]
    #      = 𝔼_q̃(ϕ)[log p(xᵢ, z)] + 𝔼_q̃(ϕ) [log |det J_f(ϕ)|] + ℍ(q̃(ϕ))

    # If f: supp(p(z | x)) → ℝ then
    # ELBO = 𝔼[log p(x, z) - log q(z)]
    #      = 𝔼[log p(x, f⁻¹(z̃)) + logabsdet(J(f⁻¹(z̃)))] + ℍ(q̃(z̃))
    #      = 𝔼[log p(x, z) - logabsdetjac(J(f(z)))] + ℍ(q̃(z̃))

    # But our `forward(q)` is using f⁻¹: ℝ → supp(p(z | x)) going forward → `+ logjac`
    _, z, logjac, _ = forward(rng, q)
    res = (logπ(z) + logjac) / num_samples

    res += (q isa TransformedDistribution) ? entropy(q.dist) : entropy(q)
    
    for i = 2:num_samples
        _, z, logjac, _ = forward(rng, q)
        res += (logπ(z) + logjac) / num_samples
    end

    return res
end

