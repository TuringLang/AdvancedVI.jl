
@testset "DynamicPPL" begin
    DynamicPPL.@model function normal(μ)
        return x ~ MvNormal(μ, I)
    end

    DynamicPPL.@model function normal_subsampled(μs; datapoints=1:size(μs, 2))
        for i in datapoints
            x ~ MvNormal(μs[:, i], I)
        end
    end

    @testset "basic" begin
        μ_true = [-2.0, 2.0]

        model = normal(μ_true)
        vi = DynamicPPL.VarInfo(model)
        vi = DynamicPPL.link!!(vi, model)

        ext = Base.get_extension(AdvancedVI, :AdvancedVIDynamicPPLExt)
        prob = ext.DynamicPPLModelLogDensityFunction(model, vi; adtype=AD)

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

        model = normal_subsampled(μs)
        vi = DynamicPPL.VarInfo(model)
        vi = DynamicPPL.link!!(vi, model)

        dataset = 1:n_data
        batchsize = 2
        subsampling = ReshufflingBatchSubsampling(dataset, batchsize)

        ext = Base.get_extension(AdvancedVI, :AdvancedVIDynamicPPLExt)
        prob = ext.DynamicPPLModelLogDensityFunction(model, vi; adtype=AD, subsampling)

        alg = KLMinRepGradProxDescent(AD; subsampling)
        d = LogDensityProblems.dimension(prob)
        q0 = FullRankGaussian(zeros(d), LowerTriangular(Matrix{Float64}(0.6 * I, d, d)))
        q, _, _ = AdvancedVI.optimize(alg, 1000, prob, q0; show_progress=false)

        Δλ0 = sum(abs2, q0.location - μ_true)
        Δλ = sum(abs2, q.location - μ_true)
        @test Δλ ≤ Δλ0 / 2
    end
end
