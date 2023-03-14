
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
        zs, ∑logjac = rand_and_logjac(rng, q, estimator.n_samples)
        
        elbo = mapreduce(+, eachcol(zs)) do zᵢ
            (logπ(zᵢ) + ∑logjac)
        end / n_samples
        -elbo
    end
    nelbo = DiffResults.value(out)
    (elbo=-nelbo,)
end
