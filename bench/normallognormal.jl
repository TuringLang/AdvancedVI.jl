struct NormalLogNormal{MX,SX,MY,SY}
    μ_x::MX
    σ_x::SX
    μ_y::MY
    Σ_y::SY
end

function LogDensityProblems.logdensity(model::NormalLogNormal, θ)
    (; μ_x, σ_x, μ_y, Σ_y) = model
    log_density_x = logpdf(LogNormal(μ_x, σ_x), θ[1])
    log_density_y = logpdf(MvNormal(μ_y, Σ_y), θ[2:end])
    return log_density_x + log_density_y
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

function normallognormal(; n_dims=10, realtype=Float64)
    μ_x = realtype(5.0)
    σ_x = realtype(0.3)
    μ_y = Fill(realtype(5.0), n_dims)
    σ_y = Fill(realtype(0.3), n_dims)
    return model = NormalLogNormal(μ_x, σ_x, μ_y, Diagonal(σ_y .^ 2))
end
