"""
$(TYPEDEF)

Black-Box Variational Inference (BBVI) with automatic differentiation
backend `AD`.

# Fields

$(TYPEDFIELDS)
"""
struct BBVI{AD} <: VariationalInference{AD}
    "Number of samples used to estimate the ELBO in each optimization step."
    samples_per_step::Int
    "Maximum number of gradient steps."
    max_iters::Int
end

function BBVI(samples_per_step::Int = 1, max_iters::Int = 1000)
    return BBVI{ADBackend()}(samples_per_step, max_iters)
end

alg_str(::BBVI) = "BBVI"
nsamples(alg::BBVI) = alg.samples_per_step
niters(alg::BBVI) = alg.max_iters

function compats(::BBVI)
    return Union{
                DiagMvNormal,
                Bijectors.TransformedDistribution{<:DiagMvNormal},
        }
end

function init(alg::BBVI, q, opt)
    samples_per_step = nsamples(alg)
    x = rand(q, samples_per_step) # Preallocating x₀
    diff_result = DiffResults.GradientResult(x)
    return (x=x, diff_result=diff_result)
end

function step!(vo::ELBO, alg::BBVI, q, logπ, state, opt)
    rand!(q, state.x) # Get initial samples from x₀
    gradobjective!(diff_result, vo, alg, logπ, q, x)
    return update!(q, x₀, diff_result, opt)
end

function update!(q, z, diff_result, opt)
    Δ = DiffResults.gradient(diff_result)
    update_mean!(q, vec(mean(Δ, dims = 2)), opt)
    update_cov!(q, Δ, z, opt)
    return nothing
end

function update_mean!(q::Bijectors.TransformedDistribution, Δ, opt)
    return update_mean!(q.dist, Δ, opt)
end

function update_mean!(q::AbstractPosteriorMvNormal, Δ, opt)
    return q.μ .+= Optimise.apply!(opt, q.μ, Δ)
end

function update_cov!(q::CholMvNormal, Δ, z, opt)
    return q.Γ .+= LowerTriangular(
        Optimise.apply!(opt, q.Γ.data, Δ * z' / size(z, 2) + inv(Diagonal(q.Γ))),
    )
end

function update_cov!(q::DiagMvNormal, Δ, z, opt)
    return q.Γ .+= Optimise.apply!(opt, q.Γ, vec(mean(Δ .* z', dims = 2)) + inv.(q.Γ))
end