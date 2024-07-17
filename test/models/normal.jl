
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

function normal_fullrank(rng::Random.AbstractRNG, realtype::Type)
    n_dims = 5

    σ0 = realtype(0.3)
    μ  = Fill(realtype(5), n_dims)
    L  = Matrix(σ0*I, n_dims, n_dims)
    Σ  = L*L' |> Hermitian

    model = TestNormal(μ, PDMat(Σ, Cholesky(L, 'L', 0)))

    TestModel(model, μ, LowerTriangular(L), n_dims, 1/σ0^2, false)
end

function normal_meanfield(rng::Random.AbstractRNG, realtype::Type)
    n_dims = 5

    σ0 = realtype(0.3)
    μ  = Fill(realtype(5), n_dims)
    σ  = Fill(σ0, n_dims)

    model = TestNormal(μ, Diagonal(σ.^2))

    L = σ |> Diagonal

    TestModel(model, μ, L, n_dims, 1/σ0^2, true)
end
