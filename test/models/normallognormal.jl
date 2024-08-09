
struct NormalLogNormal{MX,SX,MY,SY}
    μ_x::MX
    σ_x::SX
    μ_y::MY
    Σ_y::SY
end

function LogDensityProblems.logdensity(model::NormalLogNormal, θ)
    @unpack μ_x, σ_x, μ_y, Σ_y = model
    return logpdf(LogNormal(μ_x, σ_x), θ[1]) + logpdf(MvNormal(μ_y, Σ_y), θ[2:end])
end

function LogDensityProblems.dimension(model::NormalLogNormal)
    return length(model.μ_y) + 1
end

function LogDensityProblems.capabilities(::Type{<:NormalLogNormal})
    return LogDensityProblems.LogDensityOrder{0}()
end

function Bijectors.bijector(model::NormalLogNormal)
    @unpack μ_x, σ_x, μ_y, Σ_y = model
    return Bijectors.Stacked(
        Bijectors.bijector.([LogNormal(μ_x, σ_x), MvNormal(μ_y, Σ_y)]),
        [1:1, 2:(1 + length(μ_y))],
    )
end

function normallognormal_fullrank(::Random.AbstractRNG, realtype::Type)
    n_y_dims = 5

    σ0 = realtype(0.3)
    μ = Fill(realtype(5.0), n_y_dims + 1)
    L = Matrix(σ0 * I, n_y_dims + 1, n_y_dims + 1)
    Σ = Hermitian(L * L')

    model = NormalLogNormal(
        μ[1], L[1, 1], μ[2:end], PDMat(Σ[2:end, 2:end], Cholesky(L[2:end, 2:end], 'L', 0))
    )
    return TestModel(model, μ, LowerTriangular(L), n_y_dims + 1, 1 / σ0^2, false)
end

function normallognormal_meanfield(::Random.AbstractRNG, realtype::Type)
    n_y_dims = 5

    σ0 = realtype(0.3)
    μ = Fill(realtype(5), n_y_dims + 1)
    σ = Fill(σ0, n_y_dims + 1)
    L = Diagonal(σ)

    model = NormalLogNormal(μ[1], σ[1], μ[2:end], Diagonal(σ[2:end] .^ 2))

    return TestModel(model, μ, L, n_y_dims + 1, 1 / σ0^2, true)
end
