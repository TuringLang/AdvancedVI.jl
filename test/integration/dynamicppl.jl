
@testset "DynamicPPL" begin
    DynamicPPL.@model function normal(μ)
        return x ~ MvNormal(μ, I)
    end

    DynamicPPL.@model function normal_subsampled()
        μ ~ MvNormal(zeros(2), 100.0 * I)
        return x ~ DynamicPPL.independent_distribution(MvNormal(μ, I))
    end

    @testset "basic" begin
        μ_true = [-2.0, 2.0]

        model = normal(μ_true)
        vi = DynamicPPL.VarInfo(model)
        vi = DynamicPPL.link!!(vi, model)

        prob = DynamicPPL.LogDensityFunction(
            model, DynamicPPL.getlogjoint_internal, vi; adtype=AD
        )

        alg = KLMinRepGradProxDescent(AD)
        d = LogDensityProblems.dimension(prob)
        q0 = FullRankGaussian(zeros(d), LowerTriangular(Matrix{Float64}(0.6 * I, d, d)))
        q, _, _ = AdvancedVI.optimize(alg, 1000, prob, q0; show_progress=false)

        Δλ0 = sum(abs2, q0.location - μ_true)
        Δλ = sum(abs2, q.location - μ_true)
        @test Δλ ≤ Δλ0 / 2
    end

    @testset "subsampling" begin
        n_data = 32
        observations = [-2.0, 2.0] .+ randn(2, n_data)
        # The weak prior makes the MAP effectively equal to the sample mean.
        μ_true = mean(observations; dims=2)[:, 1]

        model = normal_subsampled() | (x=observations,)

        batchsize = 2
        subsampling = ReshufflingBatchSubsampling(1:n_data, batchsize)
        make_prob = batch -> DynamicPPL.subsample(model, batch, n_data)
        prob = make_prob(1:n_data)

        batch = 1:batchsize
        d = LogDensityProblems.dimension(prob)
        θ = zeros(d)
        logdensity = LogDensityProblems.logdensity(make_prob(batch), θ)
        expected =
            logpdf(MvNormal(zeros(2), 100.0 * I), θ) +
            n_data / batchsize * sum(i -> logpdf(MvNormal(θ, I), observations[:, i]), batch)
        @test logdensity ≈ expected

        alg = KLMinRepGradProxDescent(AD; subsampling)
        q0 = FullRankGaussian(zeros(d), LowerTriangular(Matrix{Float64}(0.6 * I, d, d)))
        q, _, _ = AdvancedVI.optimize(alg, 1000, make_prob, q0; show_progress=false)

        Δλ0 = sum(abs2, q0.location - μ_true)
        Δλ = sum(abs2, q.location - μ_true)
        @test Δλ ≤ Δλ0 / 2
    end
end
