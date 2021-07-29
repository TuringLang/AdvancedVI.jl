"""
$(TYPEDEF)

"Automatic Differentiation Variational Inference" (ADVI) with automatic differentiation
backend `AD`.

As described in [^ADVI16]

# Fields

$(TYPEDFIELDS)

[^ADVI16]:  Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, David M. Blei (2016), [Automatic Differentiation Variational Inference](https://arxiv.org/abs/1603.00788)
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
samples_per_step(alg::ADVI) = alg.samples_per_step
maxiters(alg::ADVI) = alg.max_iters

function compats(::ADVI)
    return Union{
        CholMvNormal,
        Bijectors.TransformedDistribution{<:CholMvNormal},
        DiagMvNormal,
        Bijectors.TransformedDistribution{<:DiagMvNormal},
    }
end

function init(rng, alg::ADVI, q, opt) # This is where the optimizer can be correctly initiated as well
    n_samples_per_step = samples_per_step(alg)
    x₀ = rand(rng, q, samples_per_step) # Preallocating x₀
    x = similar(x₀) # Preallocating x
    diff_result = DiffResults.GradientResult(x)
    return (x₀=x₀, x=x, diff_result=diff_result)
end

function step!(rng, ::ELBO, alg::ADVI, q, logπ, state, opt)
    randn!(rng, state.x₀) # Get initial samples from x₀
    reparametrize!(state.x, q, state.x₀)
    f(X) =
        sum(eachcol(X)) do x
            return evaluate(logπ, q, x)
        end
    grad!(state.diff_result, f, state.x, alg)
    return update!(alg, q, state, opt)
end

function update!(alg::ADVI, q, state, opt)
    Δ = DiffResults.gradient(state.diff_result)
    update_mean!(q, vec(mean(Δ; dims=2)), opt)
    update_cov!(alg, q, Δ, state, opt)
    return q
end

function update_cov!(alg::ADVI, q::Bijectors.TransformedDistribution, Δ, state, opt)
    return update_cov!(alg, q.dist, Δ, state, opt)
end

if VERSION < v"1.5.0"
    function update_cov!(::ADVI, q::CholMvNormal, Δ, state, opt)
        return q.Γ .+= LowerTriangular(
            Optimise.apply!(
                opt, q.Γ.data, Δ * state.x₀' / size(state.x₀, 2) + inv(Diagonal(q.Γ.data))
            ),
        )
    end
else
    function update_cov!(::ADVI, q::CholMvNormal, Δ, state, opt)
        return q.Γ .+= LowerTriangular(
            Optimise.apply!(
                opt, q.Γ.data, Δ * state.x₀' / size(state.x₀, 2) + inv(Diagonal(q.Γ))
            ),
        )
    end
end

function update_cov!(::ADVI, q::DiagMvNormal, Δ, state, opt)
    return q.Γ .+= Optimise.apply!(opt, q.Γ, vec(mean(Δ .* state.x₀; dims=2)) + inv.(q.Γ))
end

Distributions.entropy(::ADVI, q) = Distributions.entropy(q)
