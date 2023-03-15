
struct ADVI <: AbstractGradientEstimator
    n_samples::Int
end

objective(::ADVI) = "ELBO"

function estimate_gradient!(
    rng::Random.AbstractRNG,
    estimator::ADVI,
    λ::Vector{<:Real},
    rebuild::Function,
    logπ::Function,
    out::DiffResults.MutableDiffResult)

    n_samples = estimator.n_samples

    grad!(ADBackend(), λ, out) do λ′
        q = rebuild(λ′)
        zs, ∑logdetjac = rand_and_logjac(rng, q, estimator.n_samples)

        𝔼logπ = mapreduce(+, eachcol(zs)) do zᵢ
            logπ(zᵢ) / n_samples
        end
        𝔼logdetjac = ∑logdetjac/n_samples

        elbo = 𝔼logπ + 𝔼logdetjac + entropy(q)
        -elbo
    end
    nelbo = DiffResults.value(out)
    (elbo=-nelbo,)
end

