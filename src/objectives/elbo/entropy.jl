
struct ClosedFormEntropy <: AbstractEntropyEstimator end

function (::ClosedFormEntropy)(q, ::AbstractMatrix)
    entropy(q)
end

skip_entropy_gradient(::ClosedFormEntropy) = false

struct MonteCarloEntropy{IsStickingTheLanding} <: AbstractEntropyEstimator end

MonteCarloEntropy() = MonteCarloEntropy{false}()

Base.show(io::IO, entropy::MonteCarloEntropy{false}) = print(io, "MonteCarloEntropy()")

"""
  Sticking the Landing Control Variate

  # Explanation

  This eatimator forms a control variate of the form of
 
    c(z)  = 𝔼-logq(z) + logq(z) = ℍ[q] - logq(z)
 
   Adding this to the closed-form entropy ELBO estimator yields:
 
     ELBO - c(z) = 𝔼logπ(z) + ℍ[q] - c(z) = 𝔼logπ(z) - logq(z),

   which has the same expectation, but lower variance when π ≈ q,
   and higher variance when π ≉ q.

   # Reference

   Roeder, Geoffrey, Yuhuai Wu, and David K. Duvenaud.
   "Sticking the landing: Simple, lower-variance gradient estimators for
   variational inference."
   Advances in Neural Information Processing Systems 30 (2017).
"""
StickingTheLandingEntropy() = MonteCarloEntropy{true}()

skip_entropy_gradient(::MonteCarloEntropy{IsStickingTheLanding}) where {IsStickingTheLanding} = IsStickingTheLanding

Base.show(io::IO, entropy::MonteCarloEntropy{true}) = print(io, "StickingTheLandingEntropy()")

function (::MonteCarloEntropy)(q, ηs::AbstractMatrix)
    n_samples = size(ηs, 2)
    mapreduce(+, eachcol(ηs)) do ηᵢ
        -logpdf(q, ηᵢ) / n_samples
    end
end

