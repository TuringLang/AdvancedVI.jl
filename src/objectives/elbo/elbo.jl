
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

Base.string(::ELBO) = "ELBO"

function ADVI(ℓπ, b⁻¹, n_samples::Int)
    ELBO(ADVIEnergy(ℓπ, b⁻¹), ClosedFormEntropy(), n_samples)
end

function estimate_gradient!(
    adbackend::AD.AbstractBackend,
    rng::Random.AbstractRNG,
    objective::ELBO,
    λ::Vector{<:Real},
    rebuild)

    n_samples = objective.n_samples
    nelbo, grad = value_and_gradient(λ; adbackend) do λ′
        q_η = rebuild(λ′)
        ηs  = rand(rng, q_η, n_samples)

        𝔼ℓ   = objective.energy_estimator(q_η, ηs)
        ℍ    = objective.entropy_estimator(q_η, ηs)
        elbo = 𝔼ℓ + ℍ
        -elbo
    end
    first(grad), (elbo=-nelbo,)
end
