
"""
    ADVI(prob, n_samples; kwargs...)

Automatic differentiation variational inference (ADVI; Kucukelbir *et al.* 2017) objective.

# Arguments
- `prob`: An object that implements the order `K == 0` `LogDensityProblems` interface.
- `n_samples`: Number of Monte Carlo samples used to estimate the ELBO. (Type `<: Int`.)

# Keyword Arguments
- `entropy`: The estimator for the entropy term. (Type `<: AbstractEntropyEstimator`; Default: ClosedFormEntropy())
- `cv`: A control variate.
- `b`: A bijector mapping the support of the base distribution to that of `prob`. (Default: `Bijectors.identity`.)

# Requirements
- ``q_{\\lambda}`` implements `rand`.
- `logdensity(prob)` must be differentiable by the selected AD backend.

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

init(advi::ADVI) = init(advi.cv)

function (advi::ADVI)(
    rng::AbstractRNG,
    q_η::ContinuousMultivariateDistribution,
    ηs ::AbstractMatrix
)
    n_samples = size(ηs, 2)
    𝔼ℓ = mapreduce(+, eachcol(ηs)) do ηᵢ
        zᵢ, logdetjacᵢ = Bijectors.with_logabsdet_jacobian(advi.b, ηᵢ)
        (advi.ℓπ(zᵢ) + logdetjacᵢ) / n_samples
    end
    ℍ  = advi.entropy(q_η, ηs)
    𝔼ℓ + ℍ
end

"""
    (advi::ADVI)(
        q_η::ContinuousMultivariateDistribution;
        rng::AbstractRNG = Random.default_rng(),
        n_samples::Int = advi.n_samples
    )

Evaluate the ELBO using the ADVI formulation.

# Arguments
- `q_η`: Variational approximation before applying a bijector (unconstrained support).
- `n_samples`: Number of Monte Carlo samples used to estimate the ELBO.

"""
function (advi::ADVI)(
    q_η::ContinuousMultivariateDistribution;
    rng::AbstractRNG = default_rng(),
    n_samples::Int = advi.n_samples
)
    ηs = rand(rng, q_η, n_samples)
    advi(rng, q_η, ηs)
end

function estimate_advi_gradient_maybe_stl!(
    rng::AbstractRNG,
    adbackend::AbstractADType,
    advi::ADVI{P, B, StickingTheLandingEntropy, CV},
    λ::Vector{<:Real},
    restructure,
    out::DiffResults.MutableDiffResult
) where {P, B, CV}
    q_η_stop = restructure(λ)
    f(λ′) = begin
        q_η = restructure(λ′)
        ηs  = rand(rng, q_η, advi.n_samples)
        -advi(rng, q_η_stop, ηs)
    end
    value_and_gradient!(adbackend, f, λ, out)
end

function estimate_advi_gradient_maybe_stl!(
    rng::AbstractRNG,
    adbackend::AbstractADType,
    advi::ADVI{P, B, <:Union{ClosedFormEntropy, FullMonteCarloEntropy}, CV},
    λ::Vector{<:Real},
    restructure,
    out::DiffResults.MutableDiffResult
) where {P, B, CV}
    f(λ′) = begin
        q_η = restructure(λ′)
        ηs  = rand(rng, q_η, advi.n_samples)
        -advi(rng, q_η, ηs)
    end
    value_and_gradient!(adbackend, f, λ, out)
end

function estimate_gradient(
    rng::AbstractRNG,
    adbackend::AbstractADType,
    advi::ADVI,
    est_state,
    λ::Vector{<:Real},
    restructure,
    out::DiffResults.MutableDiffResult
)
    estimate_advi_gradient_maybe_stl!(
        rng, adbackend, advi, λ, restructure, out)
    nelbo = DiffResults.value(out)
    stat  = (elbo=-nelbo,)

    est_state, stat′ = update(advi.cv, est_state)
    stat = !isnothing(stat′) ? merge(stat′, stat) : stat 

    out, est_state, stat
end
