
struct UnconstrDist{D<:ContinuousMultivariateDistribution}
    dist::D
end

function LogDensityProblems.logdensity(model::UnconstrDist, x)
    return logpdf(model.dist, x)
end

function LogDensityProblems.logdensity_and_gradient(model::UnconstrDist, θ)
    return (
        LogDensityProblems.logdensity(model, θ),
        ForwardDiff.gradient(Base.Fix1(LogDensityProblems.logdensity, model), θ),
    )
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

function normal(; n_dims=10, realtype=Float64)
    μ = fill(realtype(5), n_dims)
    Σ = Diagonal(ones(realtype, n_dims))
    return UnconstrDist(MvNormal(μ, Σ))
end
