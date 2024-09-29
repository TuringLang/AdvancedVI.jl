
struct NormalLogNormal{MX,SX,MY,SY}
    μ_x::MX
    σ_x::SX
    μ_y::MY
    Σ_y::SY
end

function LogDensityProblems.logdensity(model::NormalLogNormal, θ)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    return logpdf(LogNormal(μ_x, σ_x), θ[1]) + logpdf(MvNormal(μ_y, Σ_y), θ[2:end])
end

function LogDensityProblems.dimension(model::NormalLogNormal)
    return length(model.μ_y) + 1
end

function LogDensityProblems.capabilities(::Type{<:NormalLogNormal})
    return LogDensityProblems.LogDensityOrder{0}()
end

function Bijectors.bijector(model::NormalLogNormal)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    return Bijectors.Stacked(
        Bijectors.bijector.([LogNormal(μ_x, σ_x), MvNormal(μ_y, Σ_y)]),
        [1:1, 2:(1 + length(μ_y))],
    )
end

function normallognormal(; fptype, adtype, family, objective, max_iter=10^3, kwargs...)
    n_dims = 10
    μ_x = fptype(5.0)
    σ_x = fptype(0.3)
    μ_y = Fill(fptype(5.0), n_dims)
    σ_y = Fill(fptype(0.3), n_dims)
    model = NormalLogNormal(μ_x, σ_x, μ_y, Diagonal(σ_y .^ 2))

    obj = variational_objective(objective; kwargs...)

    d = LogDensityProblems.dimension(model)
    q = variational_standard_mvnormal(fptype, d, family)

    b = Bijectors.bijector(model)
    binv = inverse(b)
    q_transformed = Bijectors.TransformedDistribution(q, binv)

    return AdvancedVI.optimize(
        model,
        obj,
        q_transformed,
        max_iter;
        adtype,
        optimizer=Optimisers.Adam(fptype(1e-3)),
        show_progress=false,
    )
end
