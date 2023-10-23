
struct TestNormal{M,S}
    μ::M
    Σ::S
end

function LogDensityProblems.logdensity(model::TestNormal, θ)
    @unpack μ, Σ = model
    logpdf(MvNormal(μ, Σ), θ)
end

function LogDensityProblems.dimension(model::TestNormal)
    length(model.μ)
end

function LogDensityProblems.capabilities(::Type{<:TestNormal})
    LogDensityProblems.LogDensityOrder{0}()
end

function normal_fullrank(realtype; rng = default_rng())
    n_dims = 5

    μ = randn(rng, realtype, n_dims)
    L = tril(I + ones(realtype, n_dims, n_dims))/2
    Σ = L*L' |> Hermitian

    model = TestNormal(μ, PDMat(Σ, Cholesky(L, 'L', 0)))

    TestModel(model, μ, L, n_dims, false)
end

function normal_meanfield(realtype; rng = default_rng())
    n_dims = 5

    μ = randn(rng, realtype, n_dims)
    σ = log.(exp.(randn(rng, realtype, n_dims)) .+ 1)

    model = TestNormal(μ, Diagonal(σ.^2))

    L = σ |> Diagonal

    TestModel(model, μ, L, n_dims, true)
end
