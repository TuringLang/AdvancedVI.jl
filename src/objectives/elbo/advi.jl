
struct ADVI{Tlogπ, B,
            EntropyEst  <: AbstractEntropyEstimator,
            ControlVar  <: Union{<: AbstractControlVariate, Nothing}} <: AbstractVariationalObjective
    ℓπ::Tlogπ
    b⁻¹::B
    entropy_estimator::EntropyEst
    control_variate::ControlVar
    n_samples::Int

    function ADVI(prob, b⁻¹, entropy_estimator, control_variate, n_samples)
        cap = LogDensityProblems.capabilities(prob)
        if cap === nothing
            throw(
                ArgumentError(
                    "The log density function does not support the LogDensityProblems.jl interface",
                ),
            )
        end
        ℓπ = Base.Fix1(LogDensityProblems.logdensity, prob)
        new{typeof(ℓπ), typeof(b⁻¹), typeof(entropy_estimator), typeof(control_variate)}(
            ℓπ, b⁻¹, entropy_estimator, control_variate, n_samples
        )
    end
end

Base.show(io::IO, advi::ADVI) =
    print(io,
          "ADVI(entropy_estimator=$(advi.entropy_estimator), " *
          "control_variate=$(advi.control_variate), " *
          "n_samples=$(advi.n_samples))")

skip_entropy_gradient(advi::ADVI) = skip_entropy_gradient(advi.entropy_estimator)

init(advi::ADVI) = init(advi.control_variate)

function ADVI(ℓπ, b⁻¹,
              entropy_estimator::AbstractEntropyEstimator,
              n_samples::Int)
    ADVI(ℓπ, b⁻¹, entropy_estimator, nothing, n_samples)
end

function ADVI(ℓπ, b⁻¹, n_samples::Int)
    ADVI(ℓπ, b⁻¹, ClosedFormEntropy(), nothing, n_samples)
end

function (advi::ADVI)(q_η::ContinuousMultivariateDistribution;
                      rng       ::AbstractRNG    = default_rng(),
                      n_samples ::Int            = advi.n_samples,
                      ηs        ::AbstractMatrix = rand(rng, q_η, n_samples),
                      q_η_entropy::ContinuousMultivariateDistribution = q_η)
    𝔼ℓ = mapreduce(+, eachcol(ηs)) do ηᵢ
        zᵢ, logdetjacᵢ = Bijectors.with_logabsdet_jacobian(advi.b⁻¹, ηᵢ)
        (advi.ℓπ(zᵢ) + logdetjacᵢ) / n_samples
    end
    ℍ  = advi.entropy_estimator(q_η_entropy, ηs)
    𝔼ℓ + ℍ
end

function estimate_gradient(
    rng::AbstractRNG,
    adback::AbstractADType,
    advi::ADVI,
    est_state,
    λ::Vector{<:Real},
    restructure,
    out::DiffResults.MutableDiffResult)

    # Gradient-stopping for computing the sticking-the-landing control variate
    q_η_stop = skip_entropy_gradient(advi.entropy_estimator) ? restructure(λ) : nothing

    grad!(adback, λ, out) do λ′
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
