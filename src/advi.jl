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
niters(alg::ADVI) = alg.max_iters

function compats(::ADVI)
    return Union{
                CholMvNormal,
                Bijectors.TransformedDistribution{<:CholMvNormal},
                DiagMvNormal,
                Bijectors.TransformedDistribution{<:DiagMvNormal},
            }
end

function init(alg::ADVI, q, opt)
    samples_per_step = nsamples(alg)
    x₀ = rand(q, samples_per_step) # Preallocating x₀
    x = similar(x₀) # Preallocating x
    diff_result = DiffResults.GradientResult(x)
    return (x₀=x₀, x=x, diff_result=diff_result)
end

function step!(vo::ELBO, alg::ADVI, q, logπ, state, opt)
    randn!(state.x₀) # Get initial samples from x₀
    reparametrize!(state.x, q, state.x₀)
    gradlogπ!(diff_result, alg, logπ, q, state.x)
    return update!(q, state, opt)
end

function update!(q, state, opt)
    Δ = DiffResults.gradient(state.diff_result)
    update_mean!(q, vec(mean(Δ, dims = 2)), opt)
    update_cov!(q, Δ, state, opt)
    return nothing
end

function update_mean!(q::Bijectors.TransformedDistribution, Δ, opt)
    return update_mean!(q.dist, Δ, opt)
end

function update_mean!(q::AbstractPosteriorMvNormal, Δ, opt)
    return q.μ .+= Optimise.apply!(opt, q.μ, Δ)
end

function update_cov!(q::CholMvNormal, Δ, state, opt)
    return q.Γ .+= LowerTriangular(
        Optimise.apply!(opt, q.Γ.data, Δ * state.x₀' / size(z, 2) + inv(Diagonal(q.Γ))),
    )
end

function update_cov!(q::DiagMvNormal, Δ, state, opt)
    return q.Γ .+= Optimise.apply!(opt, q.Γ, vec(mean(Δ .* state.x₀', dims = 2)) + inv.(q.Γ))
end