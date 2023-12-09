
@testset "interface LocationScale" begin
    @testset "$(string(covtype)) $(basedist) $(realtype)" for
        basedist = [:gaussian],
        covtype  = [:meanfield, :fullrank],
        realtype = [Float32,     Float64]

        n_dims       = 10
        n_montecarlo = 1000_000

        μ = randn(realtype, n_dims)
        L = if covtype == :fullrank
	    tril(I + ones(realtype, n_dims, n_dims)/2) |> LowerTriangular
        else
            Diagonal(ones(realtype, n_dims))
        end
        Σ = L*L'

        q = if covtype == :fullrank  && basedist == :gaussian
            FullRankGaussian(μ, L)
        elseif covtype == :meanfield && basedist == :gaussian
            MeanFieldGaussian(μ, L)
        end
        q_true = if basedist == :gaussian
            MvNormal(μ, Σ)
        end

        @testset "logpdf" begin
            z = rand(q)
            @test eltype(z)             == realtype
            @test logpdf(q, z)          ≈  logpdf(q_true, z)  rtol=realtype(1e-2)
            @test eltype(logpdf(q, z))  == realtype 
        end

        @testset "entropy" begin
            @test eltype(entropy(q)) == realtype
            @test entropy(q)         ≈ entropy(q_true)
        end

        @testset "statistics" begin
            @testset "mean" begin
                 @test eltype(mean(q)) == realtype
                 @test mean(q)         == μ
            end
            @testset "var" begin
                 @test eltype(var(q)) == realtype
		 @test var(q)         ≈  Diagonal(Σ)
            end
            @testset "cov" begin
                 @test eltype(cov(q)) == realtype
		 @test cov(q)         ≈  Σ
            end
        end

        @testset "sampling" begin
            @testset "rand" begin
                z_samples  = mapreduce(x -> rand(q), hcat, 1:n_montecarlo)
                @test eltype(z_samples) == realtype
                @test dropdims(mean(z_samples, dims=2), dims=2) ≈ μ       rtol=realtype(1e-2)
                @test dropdims(var(z_samples, dims=2),  dims=2) ≈ diag(Σ) rtol=realtype(1e-2)
                @test cov(z_samples, dims=2)                    ≈ Σ       rtol=realtype(1e-2)
            end

            @testset "rand batch" begin
                z_samples  = rand(q, n_montecarlo)
                @test eltype(z_samples) == realtype
                @test dropdims(mean(z_samples, dims=2), dims=2) ≈ μ       rtol=realtype(1e-2)
                @test dropdims(var(z_samples, dims=2),  dims=2) ≈ diag(Σ) rtol=realtype(1e-2)
                @test cov(z_samples, dims=2)                    ≈ Σ       rtol=realtype(1e-2)
            end

            @testset "rand!" begin
                z_samples = Array{realtype}(undef, n_dims, n_montecarlo)
                rand!(q, z_samples)
                @test dropdims(mean(z_samples, dims=2), dims=2) ≈ μ       rtol=realtype(1e-2)
                @test dropdims(var(z_samples, dims=2),  dims=2) ≈ diag(Σ) rtol=realtype(1e-2)
                @test cov(z_samples, dims=2)                    ≈ Σ       rtol=realtype(1e-2)
            end
        end
    end

    @testset "Diagonal destructure" for
        n_dims = 10
        μ      = zeros(n_dims)
        L      = ones(n_dims)
        q      = MeanFieldGaussian(μ, L |> Diagonal)
        λ, re  = Optimisers.destructure(q)

        @test length(λ) == 2*n_dims
        @test q         == re(λ)
    end
end
