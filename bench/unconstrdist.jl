
using Distributions, Enzyme, DifferentiationInterface, LinearAlgebra, LogDensityProblems

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

function f(x, aux)
    LogDensityProblems.logdensity(aux , x)
end

function main()
    n_dims = 10
    x      = randn(10)

    for fptype in [Float32, Float64],
        aux in [
            UnconstrDist(MvNormal(fill(fptype(5), n_dims), Diagonal(ones(fptype, n_dims)))),
            UnconstrDist(MvNormal(fill(fptype(5), n_dims), ones(fptype, n_dims))),
            UnconstrDist(MvNormal(fill(fptype(5), n_dims), I)),
        ]
        ∇x = zeros(n_dims)
        _, y = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal, true),
            Enzyme.Const(f),
            Enzyme.Active,
            Enzyme.Duplicated(x, ∇x),
            Enzyme.Const(aux),
        )
        println(y)
    end
end
