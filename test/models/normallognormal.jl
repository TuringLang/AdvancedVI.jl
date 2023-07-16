
struct NormalLogNormal{MX,SX,MY,SY}
    μ_x::MX
    σ_x::SX
    μ_y::MY
    Σ_y::SY
end

function LogDensityProblems.logdensity(model::NormalLogNormal, θ)
    @unpack μ_x, σ_x, μ_y, Σ_y = model
    logpdf(LogNormal(μ_x, σ_x), θ[1]) + logpdf(MvNormal(μ_y, Σ_y), θ[2:end])
end

function LogDensityProblems.dimension(model::NormalLogNormal)
    length(model.μ_y) + 1
end

function LogDensityProblems.capabilities(::Type{<:NormalLogNormal})
    LogDensityProblems.LogDensityOrder{0}()
end

function Bijectors.bijector(model::NormalLogNormal)
    @unpack μ_x, σ_x, μ_y, Σ_y = model
    Bijectors.Stacked(
        Bijectors.bijector.([LogNormal(μ_x, σ_x), MvNormal(μ_y, Σ_y)]),
        [1:1, 2:1+length(μ_y)])
end

function normallognormal_fullrank(realtype; rng = default_rng())
    n_dims = 5

    μ_x  = randn(rng, realtype)
    σ_x  = ℯ
    μ_y  = randn(rng, realtype, n_dims)
    L₀_y = sample_cholesky(rng, n_dims)
    ϵ    = eps(realtype)*10
    Σ_y  = (L₀_y*L₀_y' + ϵ*I) |> Hermitian

    model = NormalLogNormal(μ_x, σ_x, μ_y, PDMats.PDMat(Σ_y))

    Σ = Matrix{realtype}(undef, n_dims+1, n_dims+1)
    Σ[1,1]         = σ_x^2
    Σ[2:end,2:end] = Σ_y
    Σ = Σ |> Hermitian

    μ = vcat(μ_x, μ_y)
    L = cholesky(Σ).L |> LowerTriangular

    TestModel(model, μ, L, n_dims+1, false)
end

function normallognormal_meanfield(realtype; rng = default_rng())
    n_dims = 5

    μ_x  = randn(rng, realtype)
    σ_x  = ℯ
    μ_y  = randn(rng, realtype, n_dims)
    σ_y  = log.(exp.(randn(rng, realtype, n_dims)) .+ 1)

    model = NormalLogNormal(μ_x, σ_x, μ_y, PDMats.PDiagMat(σ_y.^2))

    μ = vcat(μ_x, μ_y)
    L = vcat(σ_x, σ_y) |> Diagonal

    TestModel(model, μ, L, n_dims+1, true)
end
