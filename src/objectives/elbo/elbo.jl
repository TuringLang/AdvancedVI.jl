
abstract type AbstractEnergyEstimator  end
abstract type AbstractEntropyEstimator end

struct ELBO{EnergyEst  <: AbstractEnergyEstimator,
            EntropyEst <: AbstractEntropyEstimator} <: AbstractVariationalObjective
    # Evidence Lower Bound
    # 
    # Jordan, Michael I., et al.
    # "An introduction to variational methods for graphical models."
    # Machine learning 37 (1999): 183-233.

    energy_estimator::EnergyEst
    entropy_estimator::EntropyEst
    n_samples::Int
end

skip_entropy_gradient(elbo::ELBO) = skip_entropy_gradient(elbo.entropy_estimator)

Base.show(io::IO, elbo::ELBO) = print(
    io,
    "ELBO(energy_estimator=$(elbo.energy_estimator), " *
    "entropy_estimator=$(elbo.entropy_estimator)), " *
    "n_samples=$(elbo.n_samples))")

function ADVI(ℓπ, b⁻¹, n_samples::Int)
    ELBO(ADVIEnergy(ℓπ, b⁻¹), ClosedFormEntropy(), n_samples)
end

function (elbo::ELBO)(q_η::ContinuousMultivariateDistribution;
                      rng = Random.default_rng(),
                      n_samples::Int = elbo.n_samples,
                      q_η_entropy::ContinuousMultivariateDistribution = q_η)
    ηs = rand(rng, q_η, n_samples)
    𝔼ℓ = elbo.energy_estimator(q_η, ηs)
    ℍ  = elbo.entropy_estimator(q_η_entropy, ηs)
    𝔼ℓ + ℍ
end

function estimate_gradient!(
    rng::Random.AbstractRNG,
    elbo::ELBO,
    λ::Vector{<:Real},
    rebuild,
    out::DiffResults.MutableDiffResult)

    # Gradient-stopping for computing the sticking-the-landing control variate
    q_η_stop = skip_entropy_gradient(elbo) ? rebuild(λ) : nothing

    grad!(ADBackend(), λ, out) do λ′
        q_η = rebuild(λ′)
        q_η_entropy = skip_entropy_gradient(elbo) ? q_η_stop : q_η
        -elbo(q_η; rng, n_samples=elbo.n_samples, q_η_entropy)
    end
    nelbo = DiffResults.value(out)
    out, (elbo=-nelbo,)
end
