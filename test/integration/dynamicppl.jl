
@testset "DynamicPPL" begin
    DynamicPPL.@model function normal(μ)
        return x ~ MvNormal(μ, I)
    end

    # Model arguments on the left of `~` are accumulated as likelihood terms.
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
        # The weak prior makes the MAP effectively equal to the sample mean.
        μ_true = mean(observations; dims=2)[:, 1]

        model = normal_minibatch(observations, n_data)
        vi = DynamicPPL.link!!(DynamicPPL.VarInfo(model), model)

        batchsize = 2
        subsampling = ReshufflingBatchSubsampling(1:n_data, batchsize)
        minibatch_model = batch -> normal_minibatch(observations[:, batch], length(batch))

        make_prob =
            (batch, scale) -> DynamicPPL.LogDensityFunction(
                minibatch_model(batch), AdvancedVI.WeightedLogJoint(scale), vi
            )
        prob = SubsampledLogDensity(make_prob(1:n_data, 1.0), make_prob, n_data)
        @test LogDensityProblems.capabilities(typeof(prob)) ==
            LogDensityProblems.LogDensityOrder{0}()

        batch = 1:batchsize
        likelihood_prob = DynamicPPL.LogDensityFunction(
            minibatch_model(batch), DynamicPPL.getloglikelihood, vi
        )
        d = LogDensityProblems.dimension(prob)
        θ = zeros(d)
        scaled = LogDensityProblems.logdensity(make_prob(batch, 2.0), θ)
        unscaled = LogDensityProblems.logdensity(make_prob(batch, 1.0), θ)
        likelihood = LogDensityProblems.logdensity(likelihood_prob, θ)
        @test scaled ≈ unscaled + likelihood

        alg = KLMinRepGradProxDescent(AD; subsampling)
        q0 = FullRankGaussian(zeros(d), LowerTriangular(Matrix{Float64}(0.6 * I, d, d)))
        q, _, _ = AdvancedVI.optimize(alg, 1000, prob, q0; show_progress=false)

        Δλ0 = sum(abs2, q0.location - μ_true)
        Δλ = sum(abs2, q.location - μ_true)
        @test Δλ ≤ Δλ0 / 2
    end
end
