
struct UnconstrDist{D <: ContinuousMultivariateDistribution}
    dist::D
end

function LogDensityProblems.logdensity(model::UnconstrDist, x)
    return logpdf(model.dist, x)
end

function LogDensityProblems.dimension(model::UnconstrDist)
    return length(model.dist)
end

function LogDensityProblems.capabilities(::Type{<:UnconstrDist})
    return LogDensityProblems.LogDensityOrder{0}()
end

function Bijectors.bijector(model::UnconstrDist)
    return identity
end

function normal(; n_dims=10, fptype=Float64)
    μ = fill(fptype(5), n_dims)
    Σ = Diagonal(ones(fptype, n_dims))
    UnconstrDist(MvNormal(μ, Σ))
end
