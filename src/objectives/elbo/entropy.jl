
struct ClosedFormEntropy <: AbstractEntropyEstimator
end

function (::ClosedFormEntropy)(q, ηs::AbstractMatrix)
    entropy(q)
end

struct MonteCarloEntropy <: AbstractEntropyEstimator
end

function (::MonteCarloEntropy)(q, ηs::AbstractMatrix)
    n_samples = size(ηs, 2)
    mapreduce(+, eachcol(ηs)) do ηᵢ
        -logpdf(q, ηᵢ) / n_samples
    end
end

