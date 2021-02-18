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
                CholMvNormal,
                # Bijectors.TransformDistribution{<:CholMvNormal},
                DiagMvNormal,
                # Bijectors.TransformedDistribution{<:DiagMvNormal},
        }
end

function init(alg::BBVI, q, opt)
    samples_per_step = nsamples(alg)
    x = rand(q, samples_per_step) # Preallocating x
    θ = to_vec(q)
    diff_result = DiffResults.GradientResult(zeros(length(θ)))
    return (x=x, θ=θ, diff_result=diff_result)
end

function step!(::ELBO, alg::BBVI, q, logπ, state, opt)
    rand!(to_dist(q, state.θ), state.x) # Get initial samples from x₀
    gradbbvi!(logπ, state, alg, q)
    return update!(alg, q, state, opt)
end

function update!(::BBVI, q, state, opt)
    state.θ .+= Optimise.apply!(opt, state.θ, DiffResults.gradient(state.diff_result))
    return nothing
end

function finish(::BBVI, q, state)
    return to_dist(q, state.θ)
end