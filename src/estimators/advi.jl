
struct ADVI{Tlogπ, B} <: AbstractGradientEstimator
    # Automatic differentiation variational inference
    # 
    # Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017).
    # Automatic differentiation variational inference.
    # Journal of machine learning research.

    ℓπ::Tlogπ
    b⁻¹::B
    n_samples::Int

    function ADVI(prob, b⁻¹::B, n_samples; kwargs...) where {B <: Bijectors.Inverse{<:Bijectors.Bijector}}
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

        zs, ∑logdetjac = Bijectors.with_logabsdet_jacobian(estimator.b⁻¹, ηs)

        𝔼logπ = mapreduce(+, eachcol(zs)) do zᵢ
            estimator.ℓπ(zᵢ) / n_samples
        end
        𝔼logdetjac = ∑logdetjac/n_samples

        elbo = 𝔼logπ + 𝔼logdetjac + entropy(q_η)
        -elbo
    end
    nelbo = DiffResults.value(out)
    (elbo=-nelbo,)
end
