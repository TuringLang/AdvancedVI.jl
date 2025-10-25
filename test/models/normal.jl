
struct TestNormal{M,S,C}
    μ::M
    Σ::S
    cap::C
end

function LogDensityProblems.logdensity(model::TestNormal, θ)
    (; μ, Σ) = model
    return logpdf(MvNormal(μ, Σ), θ)
end

function LogDensityProblems.logdensity_and_gradient(model::TestNormal, θ)
    return (
        LogDensityProblems.logdensity(model, θ),
        ForwardDiff.gradient(Base.Fix1(LogDensityProblems.logdensity, model), θ),
    )
end

function LogDensityProblems.logdensity_gradient_and_hessian(model::TestNormal, θ)
    return (
        LogDensityProblems.logdensity(model, θ),
        ForwardDiff.gradient(Base.Fix1(LogDensityProblems.logdensity, model), θ),
        ForwardDiff.hessian(Base.Fix1(LogDensityProblems.logdensity, model), θ),
    )
end

function LogDensityProblems.dimension(model::TestNormal)
    return length(model.μ)
end

function LogDensityProblems.capabilities(::Type{TestNormal{M,S,C}}) where {M,S,C}
    return C()
end

function normal_fullrank(rng::Random.AbstractRNG, realtype::Type; capability::Int=1)
    n_dims = 5

    σ0 = realtype(0.3)
    μ = Fill(realtype(5), n_dims)
    L = Matrix(σ0 * I, n_dims, n_dims)
    Σ = Hermitian(L * L')

    cap = if capability == 1
        LogDensityProblems.LogDensityOrder{1}()
    elseif capability == 2
        LogDensityProblems.LogDensityOrder{2}()
    else
        LogDensityProblems.LogDensityOrder{0}()
    end
    model = TestNormal(μ, PDMat(Σ, Cholesky(L, 'L', 0)), cap)

    return TestModel(model, μ, LowerTriangular(L), n_dims, 1 / σ0^2, false)
end

function normal_meanfield(rng::Random.AbstractRNG, realtype::Type; capability::Int=1)
    n_dims = 5

    σ0 = realtype(0.3)
    μ = Fill(realtype(5), n_dims)
    σ = Fill(σ0, n_dims)

    cap = if capability == 1
        LogDensityProblems.LogDensityOrder{1}()
    elseif capability == 2
        LogDensityProblems.LogDensityOrder{2}()
    else
        LogDensityProblems.LogDensityOrder{0}()
    end
    model = TestNormal(μ, Diagonal(σ .^ 2), cap)

    L = Diagonal(σ)

    return TestModel(model, μ, L, n_dims, 1 / σ0^2, true)
end
