
@testset "DynamicPPL" begin
    DynamicPPL.@model function normal(μ)
        return x ~ MvNormal(μ, I)
    end

    # `μ` is the latent parameter being inferred from observations stored in
    # `obs_batch`. The data observations land in `LogLikelihoodAccumulator`,
    # which is what the SG-correction scale multiplies — verifying the
    # minibatch correction actually exercises the likelihood path.
    DynamicPPL.@model function normal_minibatch(obs_batch, N)
        μ ~ MvNormal(zeros(size(obs_batch, 1)), 100.0 * I)
        for i in 1:N
            obs_batch[:, i] ~ MvNormal(μ, I)
        end
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
        # MAP target — q converges to the sample mean (weak prior is negligible
        # against `n_data` likelihood contributions).
        μ_true = mean(observations; dims=2)[:, 1]

        model = normal_minibatch(observations, n_data)
        vi = DynamicPPL.link!!(DynamicPPL.VarInfo(model), model)

        batchsize = 2
        subsampling = ReshufflingBatchSubsampling(1:n_data, batchsize)
        minibatch_model = batch -> normal_minibatch(observations[:, batch], length(batch))

        make_prob =
            (batch, scale) -> DynamicPPL.LogDensityFunction(
                minibatch_model(batch),
                AdvancedVI.WeightedLogJoint(scale),
                vi;
                adtype=AD,
            )
        prob = SubsampledLogDensity(make_prob(1:n_data, 1.0), make_prob, n_data)

        alg = KLMinRepGradProxDescent(AD; subsampling)
        d = LogDensityProblems.dimension(prob)
        q0 = FullRankGaussian(zeros(d), LowerTriangular(Matrix{Float64}(0.6 * I, d, d)))
        q, _, _ = AdvancedVI.optimize(alg, 1000, prob, q0; show_progress=false)

        Δλ0 = sum(abs2, q0.location - μ_true)
        Δλ = sum(abs2, q.location - μ_true)
        @test Δλ ≤ Δλ0 / 2

        @test_throws ArgumentError SubsampledLogDensity(prob.prob, make_prob, 0)
        @test_throws ArgumentError AdvancedVI.subsample(prob, 1:(n_data + 1))
    end
end
