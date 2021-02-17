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

function ADVI(samples_per_step::Int = 1, max_iters::Int = 1000)
    return ADVI{ADBackend()}(samples_per_step, max_iters)
end

alg_str(::ADVI) = "ADVI"
nsamples(alg::ADVI) = alg.samples_per_step

function step!(alg::ADVI, q, logπ, x₀, x, diff_result, opt)
    randn!(x₀) # Get initial samples from x₀
    reparametrize!(x, q, x₀)
    gradlogπ!(diff_result, alg, logπ, q, x)
    return update!(q, x₀, diff_result, opt)
end

function update!(q, z, diff_result, opt)
    Δ = gradient(diff_result)
    update_mean!(q, mean(Δ, dims = 2), opt)
    update_cov!(q, Δ, z, opt)
    return nothing
end

function update_mean!(q::TransformedDistribution, Δ, opt)
    return update_mean!(q.dist, Δ, opt)
end

function update_mean!(q::AbstractPosterior, Δ, opt)
    return q.μ .-= Optimise.apply!(opt, q.μ, Δ)
end

function update_cov!(q::CholMvNormal, Δ, z, opt)
    return q.Γ .-= LowerTriangular(
        Optimise.apply!(opt, q.Γ, Δ * z' / size(z, 2) - inv(Diagonal(q.Γ))),
    )
end

function update_cov!(q::DiagMvNormal, Δ, z, opt)
    return q.Γ .-= Optimise.apply!(opt, q.Γ, vec(mean(Δ .* z', dims = 2)) - inv.(q.Γ))
end