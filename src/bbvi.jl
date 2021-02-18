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
    x = rand(q, samples_per_step) # Preallocating x
    θ = to_vec(q)
    Δ = zeros(length(θ), samples_per_step)
    diff_result = DiffResults.GradientResult(x)
    return (x=x, θ=θ, diff_result=diff_result)
end

function step!(::ELBO, alg::BBVI, q, logπ, state, opt)
    rand!(q, state.x) # Get initial samples from x₀
    gradlogq!(state, alg, q)
    Δ = DiffResults.gradient(state.diff_result)
    state.Δ .= vec(mean(1:nsamples(alg); dims=2) do i
        @views Δ[:, i] * (logπ(x[:, i]) - logpdf(q, x[:, i]))
    end) 
    return update!(alg, q, state, opt)
end

function update!(::BBVI, q, state, opt)
    q.θ .+= Optimise.apply!(opt, state.θ, state.Δ)
    return nothing
end

function finish(::BBVI, q, state)
    return to_dist(q, state.θ)
end