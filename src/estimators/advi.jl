
struct ADVI{Tlogπ, B} <: AbstractGradientEstimator
    # Automatic differentiation variational inference
    # 
    # Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017).
    # Automatic differentiation variational inference.
    # Journal of machine learning research.

    ℓπ::Tlogπ
    b⁻¹::B
    n_samples::Int

    function ADVI(prob, b⁻¹, n_samples; kwargs...)
        # Could check whether the support of b⁻¹ and ℓπ match
        cap = LogDensityProblems.capabilities(prob)
        if cap === nothing
            throw(
                ArgumentError(
                    "The log density function does not support the LogDensityProblems.jl interface",
                ),
            )
        end
        ℓπ = Base.Fix1(LogDensityProblems.logdensity, prob)
        new{typeof(ℓπ), typeof(b⁻¹)}(ℓπ, b⁻¹, n_samples)
    end
end

ADVI(prob, n_samples; kwargs...) = ADVI(prob, identity, n_samples; kwargs...)

objective(::ADVI) = "ELBO"

function estimate_gradient!(
    rng::Random.AbstractRNG,
    estimator::ADVI,
    λ::Vector{<:Real},
    rebuild,
    out::DiffResults.MutableDiffResult)

    n_samples = estimator.n_samples

    grad!(ADBackend(), λ, out) do λ′
        q_η = rebuild(λ′)
        ηs  = rand(rng, q_η, estimator.n_samples)

        𝔼ℓ = mapreduce(+, eachcol(ηs)) do ηᵢ
            zᵢ, logdetjacᵢ = Bijectors.with_logabsdet_jacobian(estimator.b⁻¹, ηᵢ)
            (estimator.ℓπ(zᵢ) + logdetjacᵢ) / n_samples
        end

        elbo = 𝔼ℓ + entropy(q_η)
        -elbo
    end
    nelbo = DiffResults.value(out)
    (elbo=-nelbo,)
end
