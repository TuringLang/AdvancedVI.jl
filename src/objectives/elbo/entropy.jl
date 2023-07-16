
struct ClosedFormEntropy <: AbstractEntropyEstimator end

function (::ClosedFormEntropy)(q, ::AbstractMatrix)
    entropy(q)
end

skip_entropy_gradient(::ClosedFormEntropy) = false

abstract type MonteCarloEntropy <: AbstractEntropyEstimator end

struct FullMonteCarloEntropy <: MonteCarloEntropy end

"""
    StickingTheLandingEntropy()

# Explanation

The STL estimator forms a control variate of the form of
 
```math
\\mathrm{CV}_{\\mathrm{STL}}\\left(z\\right) =
  \\mathbb{E}\\left[ -\\log q\\left(z\\right) \\right]
  + \\log q\\left(z\\right) = \\mathbb{H}\\left(q_{\\lambda}\\right) + \\log q_{\\lambda}\\left(z\\right),
```
where, for the score term, the gradient is stopped from propagating.
 
Adding this to the closed-form entropy ELBO estimator yields the STL estimator:
```math
\\begin{aligned}
  \\widehat{\\mathrm{ELBO}}_{\\mathrm{STL}}\\left(\\lambda\\right)
    &\\triangleq \\mathbb{E}\\left[ \\log \\pi \\left(z\\right) \\right] - \\log q_{\\lambda} \\left(z\\right) \\\\
    &= \\mathbb{E}\\left[ \\log \\pi\\left(z\\right) \\right] 
      + \\mathbb{H}\\left(q_{\\lambda}\\right) - \\mathrm{CV}_{\\mathrm{STL}}\\left(z\\right) \\\\
    &= \\widehat{\\mathrm{ELBO}}\\left(\\lambda\\right)
      - \\mathrm{CV}_{\\mathrm{STL}}\\left(z\\right),
\\end{aligned}
```
which has the same expectation, but lower variance when ``\\pi \\approx q_{\\lambda}``,
and higher variance when ``\\pi \\not\\approx q_{\\lambda}``.

# Reference
1. Roeder, G., Wu, Y., & Duvenaud, D. K. (2017). Sticking the landing: Simple, lower-variance gradient estimators for variational inference. Advances in Neural Information Processing Systems, 30.
"""

struct StickingTheLandingEntropy <: MonteCarloEntropy end

function (::MonteCarloEntropy)(q, ηs::AbstractMatrix)
    n_samples = size(ηs, 2)
    mapreduce(+, eachcol(ηs)) do ηᵢ
        -logpdf(q, ηᵢ) / n_samples
    end
end

