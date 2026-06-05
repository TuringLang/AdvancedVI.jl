
@testset "DynamicPPL" begin
    DynamicPPL.@model function normal(μ)
        return x ~ MvNormal(μ, I)
    end

    DynamicPPL.@model function normal_minibatch(μs_batch, N)
        for i in 1:N
            x ~ MvNormal(μs_batch[:, i], I)
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
        μs = 3 * randn(2, n_data)
        μ_true = mean(μs; dims=2)[:, 1]

        model = normal_minibatch(μs, n_data)
        vi = DynamicPPL.link!!(DynamicPPL.VarInfo(model), model)

        batchsize = 2
        subsampling = ReshufflingBatchSubsampling(1:n_data, batchsize)
        minibatch_model = batch -> normal_minibatch(μs[:, batch], length(batch))

        make_prob = (batch, scale) -> DynamicPPL.LogDensityFunction(
            minibatch_model(batch), AdvancedVI.WeightedLogJoint(scale), vi; adtype=AD
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
        @test_throws ArgumentError AdvancedVI.with_batch(prob, 1:(n_data + 1))
    end
end
