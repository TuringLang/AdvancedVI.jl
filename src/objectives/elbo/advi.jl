
"""
    ADVI

Automatic differentiation variational inference (ADVI; Kucukelbir *et al.* 2017) objective.

# Requirements
- ``q_{\\lambda}`` implements `rand`.
- ``\\pi`` must be differentiable

Depending on the options, additional requirements on ``q_{\\lambda}`` may apply.
"""
struct ADVI{Tlogπ, B,
            EntropyEst <: AbstractEntropyEstimator,
            ControlVar <: Union{<: AbstractControlVariate, Nothing}} <: AbstractVariationalObjective
    ℓπ::Tlogπ
    b::B
    entropy::EntropyEst
    cv::ControlVar
    n_samples::Int

    function ADVI(prob, n_samples::Int;
                  entropy::AbstractEntropyEstimator = ClosedFormEntropy(),
                  cv::Union{<:AbstractControlVariate, Nothing} = nothing,
                  b = Bijectors.identity)
        cap = LogDensityProblems.capabilities(prob)
        if cap === nothing
            throw(
                ArgumentError(
                    "The log density function does not support the LogDensityProblems.jl interface",
                ),
            )
        end
        ℓπ = Base.Fix1(LogDensityProblems.logdensity, prob)
        new{typeof(ℓπ), typeof(b), typeof(entropy), typeof(cv)}(ℓπ, b, entropy, cv, n_samples)
    end
end

Base.show(io::IO, advi::ADVI) =
    print(io, "ADVI(entropy=$(advi.entropy), cv=$(advi.cv), n_samples=$(advi.n_samples))")

skip_entropy_gradient(advi::ADVI) = skip_entropy_gradient(advi.entropy)

init(advi::ADVI) = init(advi.cv)

function (advi::ADVI)(q_η::ContinuousMultivariateDistribution;
                      rng       ::AbstractRNG    = default_rng(),
                      n_samples ::Int            = advi.n_samples,
                      ηs        ::AbstractMatrix = rand(rng, q_η, n_samples),
                      q_η_entropy::ContinuousMultivariateDistribution = q_η)
    𝔼ℓ = mapreduce(+, eachcol(ηs)) do ηᵢ
        zᵢ, logdetjacᵢ = Bijectors.with_logabsdet_jacobian(advi.b, ηᵢ)
        (advi.ℓπ(zᵢ) + logdetjacᵢ) / n_samples
    end
    ℍ  = advi.entropy(q_η_entropy, ηs)
    𝔼ℓ + ℍ
end

function estimate_gradient(
    rng::AbstractRNG,
    adbackend::AbstractADType,
    advi::ADVI,
    est_state,
    λ::Vector{<:Real},
    restructure,
    out::DiffResults.MutableDiffResult)

    # Gradient-stopping for computing the sticking-the-landing control variate
    q_η_stop = skip_entropy_gradient(advi.entropy) ? restructure(λ) : nothing

    grad!(adbackend, λ, out) do λ′
        q_η = restructure(λ′)
        q_η_entropy = skip_entropy_gradient(advi.entropy) ? q_η_stop : q_η
        -advi(q_η; rng, q_η_entropy)
    end
    nelbo = DiffResults.value(out)
    stat  = (elbo=-nelbo,)

    est_state, stat′ = update(advi.cv, est_state)
    stat = !isnothing(stat′) ? merge(stat′, stat) : stat 

    out, est_state, stat
end
