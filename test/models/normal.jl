
struct TestMvNormal{M,S}
    μ::M
    Σ::S
end

function LogDensityProblems.logdensity(model::TestMvNormal, θ)
    @unpack μ, Σ = model
    logpdf(MvNormal(μ, Σ), θ)
end

function LogDensityProblems.dimension(model::TestMvNormal)
    length(model.μ)
end

function LogDensityProblems.capabilities(::Type{<:TestMvNormal})
    LogDensityProblems.LogDensityOrder{0}()
end

function Bijectors.bijector(model::TestMvNormal)
    identity
end

function normal_fullrank(realtype; rng = default_rng())
    n_dims = 5

    μ  = randn(rng, realtype, n_dims)
    L₀ = sample_cholesky(rng, n_dims)
    ϵ  = eps(realtype)*10
    Σ  = (L₀*L₀' + ϵ*I) |> Hermitian

    Σ_chol = cholesky(Σ)
    model  = TestMvNormal(μ, PDMats.PDMat(Σ, Σ_chol))

    L = Σ_chol.L |> LowerTriangular

    TestModel(model, μ, L, n_dims, false)
end

function normal_meanfield(realtype; rng = default_rng())
    n_dims = 5

    μ = randn(rng, realtype, n_dims)
    σ = log.(exp.(randn(rng, realtype, n_dims)) .+ 1)

    model = TestMvNormal(μ, PDMats.PDiagMat(σ))

    L = σ |> Diagonal

    TestModel(model, μ, L, n_dims, true)
end
