
@testset "KLMinWassFwdBwd" begin
    begin
        modelstats = normal_meanfield(Random.default_rng(), Float64; capability=2)
        (; model, n_dims, μ_true, L_true) = modelstats

        alg = KLMinWassFwdBwd(; n_samples=10, stepsize=1e-3)
        L0 = LowerTriangular(Matrix{Float64}(I, n_dims, n_dims))
        q0 = FullRankGaussian(zeros(Float64, n_dims), L0)

        @testset "callback" begin
            T = 10
            callback(; iteration, kwargs...) = (iteration_check=iteration,)
            _, info, _ = optimize(alg, T, model, q0; callback, show_progress=PROGRESS)
            @test [i.iteration_check for i in info] == 1:T
        end

        @testset "estimate_objective" begin
            q_true = FullRankGaussian(μ_true, LowerTriangular(Matrix(L_true)))

            obj_est = estimate_objective(alg, q_true, model)
            @test isfinite(obj_est)

            obj_est = estimate_objective(alg, q_true, model; n_samples=10^5)
            @test obj_est ≈ 0 atol=1e-2
        end

        @testset "determinism" begin
            seed = (0x38bef07cf9cc549d)
            rng = StableRNG(seed)
            T = 10

            q_avg, _, _ = optimize(rng, alg, T, model, q0; show_progress=PROGRESS)
            μ = q_avg.location
            L = q_avg.scale

            rng_repl = StableRNG(seed)
            q_avg, _, _ = optimize(rng_repl, alg, T, model, q0; show_progress=PROGRESS)
            μ_repl = q_avg.location
            L_repl = q_avg.scale
            @test μ == μ_repl
            @test L == L_repl
        end
    end

    begin
        alg = KLMinWassFwdBwd(; n_samples=10, stepsize=1.0)

        @testset "error low capability" begin
            modelstats = normal_meanfield(Random.default_rng(), Float64; capability=0)
            (; model, n_dims) = modelstats

            L0 = LowerTriangular(Matrix{Float64}(I, n_dims, n_dims))
            q0 = FullRankGaussian(zeros(Float64, n_dims), L0)
            @test_throws "first-order" optimize(alg, 1, model, q0)
        end
    end

    @testset "type stability type=$(realtype), capability=$(capability)" for realtype in [
            Float64, Float32
        ],
        capability in [1, 2]

        modelstats = normal_meanfield(Random.default_rng(), realtype; capability)
        (; model, μ_true, L_true, n_dims, strong_convexity, is_meanfield) = modelstats

        alg = KLMinWassFwdBwd(; n_samples=10, stepsize=1e-3)
        T = 10

        L0 = LowerTriangular(Matrix{realtype}(I, n_dims, n_dims))
        q0 = FullRankGaussian(zeros(realtype, n_dims), L0)

        q, _, _ = optimize(alg, T, model, q0; show_progress=PROGRESS)

        @test eltype(q.location) == eltype(μ_true)
        @test eltype(q.scale) == eltype(L_true)
    end

    @testset "convergence capability=$(capability)" for capability in [1, 2]
        modelstats = normal_meanfield(Random.default_rng(), Float64; capability)
        (; model, μ_true, L_true, n_dims, strong_convexity, is_meanfield) = modelstats

        T = 1000
        alg = KLMinWassFwdBwd(; n_samples=10, stepsize=1e-3)

        q_avg, _, _ = optimize(alg, T, model, q0; show_progress=PROGRESS)

        Δλ0 = sum(abs2, q0.location - μ_true) + sum(abs2, q0.scale - L_true)
        Δλ = sum(abs2, q_avg.location - μ_true) + sum(abs2, q_avg.scale - L_true)

        @test Δλ ≤ 0.1*Δλ0
    end

    @testset "subsampling" begin
        n_data = 8

        @testset "estimate_objective batchsize=$(batchsize)" for batchsize in [1, 3, 4]
            modelstats = subsamplednormal(Random.default_rng(), n_data)
            (; model, n_dims, μ_true, L_true) = modelstats

            L0 = LowerTriangular(Matrix{Float64}(I, n_dims, n_dims))
            q0 = FullRankGaussian(zeros(Float64, n_dims), L0)

            subsampling = ReshufflingBatchSubsampling(1:n_data, batchsize)
            alg = KLMinWassFwdBwd(; n_samples=10, stepsize=1e-3)
            alg_sub = KLMinWassFwdBwd(; n_samples=10, stepsize=1e-3, subsampling)

            obj_full = estimate_objective(alg, q0, model; n_samples=10^5)
            obj_sub = estimate_objective(alg_sub, q0, model; n_samples=10^5)
            @test obj_full ≈ obj_sub rtol=0.1
        end

        @testset "determinism" begin
            seed = (0x38bef07cf9cc549d)
            rng = StableRNG(seed)

            modelstats = subsamplednormal(Random.default_rng(), n_data)
            (; model, n_dims, μ_true, L_true) = modelstats

            L0 = LowerTriangular(Matrix{Float64}(I, n_dims, n_dims))
            q0 = FullRankGaussian(zeros(Float64, n_dims), L0)

            T = 10
            batchsize = 3
            subsampling = ReshufflingBatchSubsampling(1:n_data, batchsize)
            alg_sub = KLMinWassFwdBwd(; n_samples=10, stepsize=1e-3, subsampling)

            q, _, _ = optimize(rng, alg_sub, T, model, q0; show_progress=PROGRESS)
            μ = q.location
            L = q.scale

            rng_repl = StableRNG(seed)
            q, _, _ = optimize(rng_repl, alg_sub, T, model, q0; show_progress=PROGRESS)
            μ_repl = q.location
            L_repl = q.scale
            @test μ == μ_repl
            @test L == L_repl
        end

        @testset "convergence capability=$(capability)" for capability in [1, 2]
            modelstats = subsamplednormal(Random.default_rng(), n_data; capability)
            (; model, n_dims, μ_true, L_true) = modelstats

            L0 = LowerTriangular(Matrix{Float64}(I, n_dims, n_dims))
            q0 = FullRankGaussian(zeros(Float64, n_dims), L0)

            T = 1000
            batchsize = 1
            subsampling = ReshufflingBatchSubsampling(1:n_data, batchsize)
            alg_sub = KLMinWassFwdBwd(; n_samples=10, stepsize=1e-2, subsampling)

            q, stats, _ = optimize(alg_sub, T, model, q0; show_progress=PROGRESS)

            Δλ0 = sum(abs2, q0.location - μ_true) + sum(abs2, q0.scale - L_true)
            Δλ = sum(abs2, q.location - μ_true) + sum(abs2, q.scale - L_true)

            @test Δλ ≤ 0.1*Δλ0
        end
    end
end
