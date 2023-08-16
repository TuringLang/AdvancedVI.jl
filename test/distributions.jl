
using ReTest
using Distributions: _logpdf

@testset "distributions" begin
    @testset "$(string(covtype)) $(basedist) $(realtype)" for
        basedist = [:gaussian],
        covtype  = [:meanfield, :fullrank],
        realtype = [Float32,     Float64]

        seed         = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng          = Philox4x(UInt64, seed, 8)
        realtype     = Float64
        ϵ            = 1f-2
        n_dims       = 10
        n_montecarlo = 1000_000

        μ  = randn(rng, realtype, n_dims)
        L₀ = randn(rng, realtype, n_dims, n_dims) |> LowerTriangular
        Σ  = if covtype == :fullrank
            Σ = (L₀*L₀' + ϵ*I) |> Hermitian
        else
            Diagonal(log.(exp.(randn(rng, realtype, n_dims)) .+ 1))
        end

        L = cholesky(Σ).L
        q = if covtype == :fullrank  && basedist == :gaussian
            VIFullRankGaussian(μ, L |> LowerTriangular)
        elseif covtype == :meanfield && basedist == :gaussian
            VIMeanFieldGaussian(μ, L |> Diagonal)
        end
        q_true = if basedist == :gaussian
            MvNormal(μ, Σ)
        end

        @testset "logpdf" begin
            z = randn(rng, realtype, n_dims)
            @test logpdf(q, z)  ≈ logpdf(q_true, z)
            @test _logpdf(q, z) ≈ _logpdf(q_true, z)
            @test eltype(logpdf(q, z))  == realtype
            @test eltype(_logpdf(q, z)) == realtype
        end

        @testset "entropy" begin
            @test eltype(entropy(q)) == realtype
            @test entropy(q)         ≈ entropy(q_true)
        end

        @testset "sampling" begin
            z_samples  = rand(rng, q, n_montecarlo)
            threesigma = L
            @test eltype(z_samples) == realtype
            @test dropdims(mean(z_samples, dims=2), dims=2) ≈ μ       rtol=realtype(1e-2)
            @test dropdims(var(z_samples, dims=2),  dims=2) ≈ diag(Σ) rtol=realtype(1e-2)
            @test cov(z_samples, dims=2)                    ≈ Σ       rtol=realtype(1e-2)
        end
    end
end
