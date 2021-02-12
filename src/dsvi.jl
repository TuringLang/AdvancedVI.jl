"""
$(TYPEDEF)

Doubly Stochastic Variational Inference (DSVI) with automatic differentiation
backend `AD`.

# Fields

$(TYPEDFIELDS)
"""
struct DSVI{AD} <: VariationalInference{AD}
    "Number of samples used to estimate the ELBO in each optimization step."
    samples_per_step::Int
    "Maximum number of gradient steps."
    max_iters::Int
end

function DSVI(samples_per_step::Int = 1, max_iters::Int = 1000)
    return DSVI{ADBackend()}(samples_per_step, max_iters)
end

alg_str(::DSVI) = "DSVI"
nsamples(alg::DSVI) = alg.samples_per_step

function step!(alg::DSVI, q, logπ, x₀, x, diff_result)
    randn!(x₀) # Get initial samples from x₀
    reparametrize!(x, q, x₀)
    gradlogπ!(diff_result, alg, logπ, q, x)
    return update!(q, x₀, diff_result, optimizer)
end

function update!(q, z, diff_result, optimizer)
    Δ = gradient(diff_result)
    update_mean!(q, mean(Δ, dims = 2), optimizer)
    return update_cov!(q, Δ, z, optimizer)
end

function update_mean!(q::TransformedDistribution, Δ, optimizer)
    return update_mean!(q.dist, Δ, optimizer)
end

function update_mean!(q::AbstractPosterior, Δ, optimizer)
    return q.μ .-= Optimise.apply!(optimizer, q.μ, Δ)
end

function update_cov!(q::CholMvNormal, Δ, z, optimizer)
    return q.Γ .-= LowerTriangular(
        Optimise.apply!(optimizer, q.Γ, Δ * z' / size(z, 2) - inv(Diagonal(q.Γ))),
    )
end

function update_cov!(q::DiagMvNormal, Δ, z, optimizer)
    return q.Γ .-= Optimise.apply!(optimizer, q.Γ, vec(mean(Δ .* z', dims = 2)) - inv.(q.Γ))
end

function updateΓ(Δ::AbstractMatrix, z::AbstractMatrix, Γ::AbstractVector)
    return vec(mean(Δ .* z, dims = 2)) - inv.(Γ)
end

function updateΓ(Δ::AbstractMatrix, z::AbstractMatrix, Γ::LowerTriangular)
    return LowerTriangular(Δ * z' / size(z, 2)) - inv(Diagonal(Γ))
end