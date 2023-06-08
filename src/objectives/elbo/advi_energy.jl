
struct ADVIEnergy{Tlogπ, B} <: AbstractEnergyEstimator
    # Automatic differentiation variational inference
    # 
    # Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017).
    # Automatic differentiation variational inference.
    # Journal of machine learning research.

    ℓπ::Tlogπ
    b⁻¹::B

    function ADVIEnergy(prob, b⁻¹)
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
        new{typeof(ℓπ), typeof(b⁻¹)}(ℓπ, b⁻¹)
    end
end

ADVIEnergy(prob) = ADVIEnergy(prob, identity)

function (energy::ADVIEnergy)(q, ηs::AbstractMatrix)
    n_samples = size(ηs, 2)
    mapreduce(+, eachcol(ηs)) do ηᵢ
        zᵢ, logdetjacᵢ = Bijectors.with_logabsdet_jacobian(energy.b⁻¹, ηᵢ)
        (energy.ℓπ(zᵢ) + logdetjacᵢ) / n_samples
    end
end
