
using ReTest
using Distributions: _logpdf

@testset "distributions" begin
    @testset "$(string(covtype)) $(basedist) $(realtype)" for
        basedist = [:gaussian],
        covtype  = [:meanfield, :fullrank],
        realtype = [Float32,     Float64]

        seed         = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
        rng          = Philox4x(UInt64, seed, 8)
        n_dims       = 10
        n_montecarlo = 1000_000

        μ = randn(rng, realtype, n_dims)
        L = if covtype == :fullrank
            sample_cholesky(rng, realtype, n_dims)
        else
            Diagonal(log.(exp.(randn(rng, realtype, n_dims)) .+ 1))
        end
        Σ = L*L'

        q = if covtype == :fullrank  && basedist == :gaussian
            VIFullRankGaussian(μ, L)
        elseif covtype == :meanfield && basedist == :gaussian
            VIMeanFieldGaussian(μ, L)
        end
        q_true = if basedist == :gaussian
            MvNormal(μ, Σ)
        end

        @testset "logpdf" begin
            z = rand(rng, q)
            @test eltype(z)             == realtype
            @test logpdf(q, z)          ≈  logpdf(q_true, z)  rtol=realtype(1e-2)
            @test _logpdf(q, z)         ≈  _logpdf(q_true, z) rtol=realtype(1e-2)
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

    @testset "Diagonal destructure" for
        n_dims = 10
        μ      = zeros(n_dims)
        L      = ones(n_dims)
        q      = VIMeanFieldGaussian(μ, L |> Diagonal)
        λ, re  = Optimisers.destructure(q)

        @test length(λ) == 2*n_dims
        @test q         == re(λ)
    end
end
