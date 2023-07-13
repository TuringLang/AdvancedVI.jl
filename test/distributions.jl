
using ReTest
using Distributions
using Distributions: _logpdf
using LinearAlgebra
using AdvancedVI: LocationScale, VIFullRankGaussian, VIMeanFieldGaussian

@testset "distributions" begin
    @testset "$(string(covtype)) Gaussian $(realtype)" for
        covtype  = [:diagonal, :fullrank],
        realtype = [Float32,     Float64]

        realtype     = Float64
        ϵ            = 1e-2
        n_dims       = 10
        n_montecarlo = 1000_000

        μ  = randn(realtype, n_dims)
        L₀ = randn(realtype, n_dims, n_dims)
        Σ  = if covtype == :fullrank
            Σ = (L₀*L₀' + ϵ*I) |> Hermitian
        else
            Diagonal(exp.(randn(realtype, n_dims)))
        end

        L = cholesky(Σ).L
        q = if covtype == :fullrank
            VIFullRankGaussian(μ, L |> LowerTriangular)
        else
            VIMeanFieldGaussian(μ, L |> Diagonal)
        end
        q_true = MvNormal(μ, Σ)

        z = randn(n_dims)
        @test logpdf(q, z)  ≈ logpdf(q_true, z)
        @test _logpdf(q, z) ≈ _logpdf(q_true, z)
        @test entropy(q)    ≈ entropy(q_true)

        z_samples  = rand(q, n_montecarlo)
        threesigma = L
        @test dropdims(mean(z_samples, dims=2), dims=2) ≈ μ       rtol=realtype(1e-2)
        @test dropdims(var(z_samples, dims=2),  dims=2) ≈ diag(Σ) rtol=realtype(1e-2)
        @test cov(z_samples, dims=2)                    ≈ Σ       rtol=realtype(1e-2)
    end
end
