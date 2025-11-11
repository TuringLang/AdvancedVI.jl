
using BenchmarkTools

struct TestQuad{S,C}
    Σ::S
    cap::C
end

function LogDensityProblems.logdensity(model::TestQuad, x)
    Σ = model.Σ
    return -x'*Σ*x/2
end

function LogDensityProblems.logdensity_and_gradient(model::TestQuad, x)
    Σ = model.Σ
    return (LogDensityProblems.logdensity(model, x), -Σ*x)
end

function LogDensityProblems.logdensity_gradient_and_hessian(model::TestQuad, x)
    Σ = model.Σ
    ℓp, ∇ℓp = LogDensityProblems.logdensity_and_gradient(model, x)
    return (ℓp, ∇ℓp, -Σ)
end

function LogDensityProblems.dimension(model::TestQuad)
    return size(model.Σ, 1)
end

function LogDensityProblems.capabilities(::Type{TestQuad{S,C}}) where {S,C}
    return C()
end

@testset "gauss_expected_grad_hess" begin
    n_samples = 10^6
    d = 2
    Σ = [2.0 -0.1; -0.1 2.0]
    q = FullRankGaussian(ones(d), LowerTriangular(diagm(fill(0.1, d))))

    # True expected gradient is E_{x ~ N(μ, 1)} -Σ x = -Σ μ
    # True expected Hessian is E_{x ~ N(μ, 1)} -Σ = -Σ
    E_∇ℓπ = -Σ*q.location
    E_∇2ℓπ = -Σ

    @testset "$(cap)-order capability" for cap in [
        LogDensityProblems.LogDensityOrder{1}(), LogDensityProblems.LogDensityOrder{2}()
    ]
        grad_buf = zeros(d)
        hess_buf = zeros(d, d)
        prob = TestQuad(Σ, cap)
        display(@benchmark AdvancedVI.gaussian_expectation_gradient_and_hessian!(
            Random.default_rng(), $q, $n_samples, $grad_buf, $hess_buf, $prob
        ))
        @test grad_buf ≈ E_∇ℓπ atol=1e-1
        @test hess_buf ≈ E_∇2ℓπ atol=1e-1
    end
end
