
struct ADVI{Tlogπ} <: AbstractGradientEstimator
    ℓπ::Tlogπ
    n_samples::Int
end

function ADVI(ℓπ, n_samples; kwargs...)
    # ADVI requires gradients of log-likelihood
    cap = LogDensityProblems.capabilities(ℓπ)
    if cap === nothing
        throw(
            ArgumentError(
                "The log density function does not support the LogDensityProblems.jl interface",
            ),
        )
    end
    ADVI(Base.Fix1(LogDensityProblems.logdensity, ℓπ), n_samples)
end

objective(::ADVI) = "ELBO"

function estimate_gradient!(
    rng::Random.AbstractRNG,
    estimator::ADVI,
    λ::Vector{<:Real},
    rebuild::Function,
    out::DiffResults.MutableDiffResult)

    n_samples = estimator.n_samples

    grad!(ADBackend(), λ, out) do λ′
        q = rebuild(λ′)
        zs, ∑logdetjac = rand_and_logjac(rng, q, estimator.n_samples)

        𝔼logπ = mapreduce(+, eachcol(zs)) do zᵢ
            estimator.ℓπ(zᵢ) / n_samples
        end
        𝔼logdetjac = ∑logdetjac/n_samples

        elbo = 𝔼logπ + 𝔼logdetjac + entropy(q)
        -elbo
    end
    nelbo = DiffResults.value(out)
    (elbo=-nelbo,)
end

