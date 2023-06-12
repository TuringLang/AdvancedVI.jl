
struct ADVI{EnergyEst   <: AbstractEnergyEstimator,
            EntropyEst  <: AbstractEntropyEstimator,
            ControlVar  <: Union{<: AbstractControlVariate, Nothing}} <: AbstractVariationalObjective
    energy_estimator::EnergyEst
    entropy_estimator::EntropyEst
    control_variate::ControlVar
    n_samples::Int
end

skip_entropy_gradient(advi::ADVI) = skip_entropy_gradient(advi.entropy_estimator)

init(advi::ADVI) = init(advi.control_variate)

Base.show(io::IO, advi::ADVI) = print(
    io,
    "ADVI(energy_estimator=$(advi.energy_estimator), " *
    "entropy_estimator=$(advi.entropy_estimator), " *
    (!isnothing(advi.control_variate) ? "control_variate=$(advi.control_variate), " : "") *
    "n_samples=$(advi.n_samples))")

function ADVI(energy_estimator::AbstractEnergyEstimator,
              entropy_estimator::AbstractEntropyEstimator,
              n_samples::Int)
    ADVI(energy_estimator, entropy_estimator, nothing, n_samples)
end

function ADVI(ℓπ, b⁻¹, n_samples::Int)
    ADVI(ADVIEnergy(ℓπ, b⁻¹), ClosedFormEntropy(), n_samples)
end

function (advi::ADVI)(q_η::ContinuousMultivariateDistribution;
                      rng       ::Random.AbstractRNG = Random.default_rng(),
                      n_samples ::Int                = advi.n_samples,
                      ηs        ::AbstractMatrix     = rand(rng, q_η, n_samples),
                      q_η_entropy::ContinuousMultivariateDistribution = q_η)
    𝔼ℓ = advi.energy_estimator(q_η, ηs)
    ℍ  = advi.entropy_estimator(q_η_entropy, ηs)
    𝔼ℓ + ℍ
end

function estimate_gradient(
    rng::Random.AbstractRNG,
    advi::ADVI,
    est_state,
    λ::Vector{<:Real},
    restructure,
    out::DiffResults.MutableDiffResult)

    # Gradient-stopping for computing the sticking-the-landing control variate
    q_η_stop = skip_entropy_gradient(advi.entropy_estimator) ? restructure(λ) : nothing

    grad!(ADBackend(), λ, out) do λ′
        q_η = restructure(λ′)
        q_η_entropy = skip_entropy_gradient(advi.entropy_estimator) ? q_η_stop : q_η
        -advi(q_η; rng, q_η_entropy)
    end
    nelbo = DiffResults.value(out)
    stat  = (elbo=-nelbo,)

    est_state, stat′ = update(advi.control_variate, est_state)
    stat = !isnothing(stat′) ? merge(stat′, stat) : stat 

    out, est_state, stat
end
