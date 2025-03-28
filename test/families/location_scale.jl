
@testset "interface LocationScale" begin
    @testset "$(string(covtype)) $(basedist) $(realtype)" for basedist in
                                                              [:gaussian, :gaussian_nonstd],
        covtype in [:meanfield, :fullrank],
        realtype in [Float32, Float64]

        n_dims = 10
        n_montecarlo = 1000_000

        location = randn(realtype, n_dims)
        scale = if covtype == :fullrank
            LowerTriangular(tril(I + ones(realtype, n_dims, n_dims) / 2))
        else
            Diagonal(ones(realtype, n_dims))
        end

        q = if covtype == :fullrank && basedist == :gaussian
            FullRankGaussian(location, scale)
        elseif covtype == :meanfield && basedist == :gaussian
            MeanFieldGaussian(location, scale)
        elseif covtype == :fullrank && basedist == :gaussian_nonstd
            MvLocationScale(location, scale, Normal(realtype(3), realtype(3)))
        elseif covtype == :meanfield && basedist == :gaussian_nonstd
            MvLocationScale(location, scale, Normal(realtype(3), realtype(3)))
        end

        q_true = if basedist == :gaussian
            MvNormal(location, scale * scale')
        elseif basedist == :gaussian_nonstd
            MvNormal(location + scale * fill(3, n_dims), 9 * scale * scale')
        end

        @testset "eltype" begin
            @test eltype(q) == realtype
        end

        @testset "logpdf" begin
            z = rand(q)
            @test logpdf(q, z) ≈ logpdf(q_true, z) rtol = realtype(1e-2)
            @test eltype(logpdf(q, z)) == realtype
        end

        @testset "entropy" begin
            @test eltype(entropy(q)) == realtype
            @test entropy(q) ≈ entropy(q_true)
        end

        @testset "length" begin
            @test length(q) == n_dims
        end

        @testset "statistics" begin
            @testset "mean" begin
                @test eltype(mean(q)) == realtype
                @test mean(q) ≈ mean(q_true)
            end
            @testset "var" begin
                @test eltype(var(q)) == realtype
                @test var(q) ≈ var(q_true)
            end
            @testset "cov" begin
                @test eltype(cov(q)) == realtype
                @test cov(q) ≈ cov(q_true)
            end
        end

        @testset "sampling" begin
            @testset "rand" begin
                z_samples = mapreduce(x -> rand(q), hcat, 1:n_montecarlo)
                @test eltype(z_samples) == realtype
                @test dropdims(mean(z_samples; dims=2); dims=2) ≈ mean(q_true) rtol = realtype(
                    1e-2
                )
                @test dropdims(var(z_samples; dims=2); dims=2) ≈ var(q_true) rtol = realtype(
                    1e-2
                )
                @test cov(z_samples; dims=2) ≈ cov(q_true) rtol = realtype(1e-2)

                z_sample_ref = rand(StableRNG(1), q)
                @test z_sample_ref ≈ rand(StableRNG(1), q)
            end

            @testset "rand batch" begin
                z_samples = rand(q, n_montecarlo)
                @test eltype(z_samples) == realtype
                @test dropdims(mean(z_samples; dims=2); dims=2) ≈ mean(q_true) rtol = realtype(
                    1e-2
                )
                @test dropdims(var(z_samples; dims=2); dims=2) ≈ var(q_true) rtol = realtype(
                    1e-2
                )
                @test cov(z_samples; dims=2) ≈ cov(q_true) rtol = realtype(1e-2)

                samples_ref = rand(StableRNG(1), q, n_montecarlo)
                @test samples_ref ≈ rand(StableRNG(1), q, n_montecarlo)
            end

            @testset "rand! AbstractVector" begin
                res = map(1:n_montecarlo) do _
                    z_sample = Array{realtype}(undef, n_dims)
                    z_sample_ret = rand!(q, z_sample)
                    (z_sample, z_sample_ret)
                end
                z_samples = mapreduce(first, hcat, res)
                z_samples_ret = mapreduce(last, hcat, res)
                @test z_samples ≈ z_samples_ret
                @test dropdims(mean(z_samples; dims=2); dims=2) ≈ mean(q_true) rtol = realtype(
                    1e-2
                )
                @test dropdims(var(z_samples; dims=2); dims=2) ≈ var(q_true) rtol = realtype(
                    1e-2
                )
                @test cov(z_samples; dims=2) ≈ cov(q_true) rtol = realtype(1e-2)

                z_sample_ref = Array{realtype}(undef, n_dims)
                rand!(StableRNG(1), q, z_sample_ref)

                z_sample = Array{realtype}(undef, n_dims)
                rand!(StableRNG(1), q, z_sample)
                @test z_sample_ref == z_sample
            end

            @testset "rand! AbstractMatrix" begin
                z_samples = Array{realtype}(undef, n_dims, n_montecarlo)
                z_samples_ret = rand!(q, z_samples)
                @test z_samples ≈ z_samples_ret
                @test dropdims(mean(z_samples; dims=2); dims=2) ≈ mean(q_true) rtol = realtype(
                    1e-2
                )
                @test dropdims(var(z_samples; dims=2); dims=2) ≈ var(q_true) rtol = realtype(
                    1e-2
                )
                @test cov(z_samples; dims=2) ≈ cov(q_true) rtol = realtype(1e-2)

                z_samples_ref = Array{realtype}(undef, n_dims, n_montecarlo)
                rand!(StableRNG(1), q, z_samples_ref)

                z_samples = Array{realtype}(undef, n_dims, n_montecarlo)
                rand!(StableRNG(1), q, z_samples)
                @test z_samples_ref ≈ z_samples
            end
        end
    end

    @testset "Diagonal destructure" begin
        n_dims = 10
        μ = zeros(n_dims)
        L = ones(n_dims)
        q = MeanFieldGaussian(μ, Diagonal(L))
        λ, re = Optimisers.destructure(q)

        @test length(λ) == 2 * n_dims
        @test q == re(λ)
    end
end
