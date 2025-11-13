
@testset "KLMinScoreGradDescent" begin
    begin
        modelstats = normal_meanfield(Random.default_rng(), Float64)
        (; model, n_dims, μ_true, L_true) = modelstats

        q0 = MeanFieldGaussian(zeros(n_dims), Diagonal(ones(n_dims)))

        @testset "basic n_samples=$(n_samples)" for n_samples in [1, 10]
            alg = KLMinScoreGradDescent(AD; n_samples, operator=ClipScale())
            T = 1
            optimize(alg, T, model, q0; show_progress=PROGRESS)
        end

        @testset "callback" begin
            alg = KLMinScoreGradDescent(AD; n_samples=10, operator=ClipScale())
            T = 10
            callback(; iteration, kwargs...) = (iteration_check=iteration,)
            _, info, _ = optimize(alg, T, model, q0; callback, show_progress=PROGRESS)
            @test [i.iteration_check for i in info] == 1:T
        end

        @testset "estimate_objective" begin
            alg = KLMinScoreGradDescent(AD; n_samples=10, operator=ClipScale())
            q_true = MeanFieldGaussian(Vector(μ_true), Diagonal(L_true))

            obj_est = estimate_objective(alg, q_true, model)
            @test isfinite(obj_est)

            obj_est = estimate_objective(alg, q_true, model; n_samples=1)
            @test isfinite(obj_est)

            obj_est = estimate_objective(alg, q_true, model; n_samples=3)
            @test isfinite(obj_est)

            obj_est = estimate_objective(alg, q_true, model; n_samples=10^5)
            @test obj_est ≈ 0 atol=1e-3
        end

        @testset "determinism" begin
            alg = KLMinScoreGradDescent(AD; n_samples=10, operator=ClipScale())

            seed = (0x38bef07cf9cc549d)
            rng = StableRNG(seed)
            T = 10

            q_out, _, _ = optimize(rng, alg, T, model, q0; show_progress=PROGRESS)
            μ = q_out.location
            L = q_out.scale

            rng_repl = StableRNG(seed)
            q_out, _, _ = optimize(rng_repl, alg, T, model, q0; show_progress=PROGRESS)
            μ_repl = q_out.location
            L_repl = q_out.scale
            @test μ == μ_repl
            @test L == L_repl
        end

        @testset "warn MvLocationScale with IdentityOperator" begin
            @test_warn "IdentityOperator" begin
                alg′ = KLMinScoreGradDescent(AD; operator=IdentityOperator())
                optimize(alg′, 1, model, q0; show_progress=false)
            end
        end
    end

    @testset "type stability realtype=$(realtype)" for realtype in [Float32, Float64]
        modelstats = normal_meanfield(Random.default_rng(), realtype)
        (; model, n_dims, μ_true, L_true) = modelstats

        T = 1
        alg = KLMinScoreGradDescent(AD; n_samples=10, operator=ClipScale())
        q0 = MeanFieldGaussian(zeros(realtype, n_dims), Diagonal(ones(realtype, n_dims)))

        q_out, info, _ = optimize(alg, T, model, q0; show_progress=PROGRESS)

        @test eltype(q_out.location) == realtype
        @test eltype(q_out.scale) == realtype
        @test typeof(first(info).elbo) == realtype
    end

    @testset "convergence" begin
        modelstats = normal_meanfield(Random.default_rng(), Float64)
        (; model, μ_true, L_true, is_meanfield) = modelstats

        T = 1000
        optimizer = Descent(1e-3)
        alg = KLMinScoreGradDescent(AD; n_samples=100, optimizer, operator=ClipScale())
        q0 = MeanFieldGaussian(zeros(n_dims), Diagonal(ones(n_dims)))

        q_out, _, _ = optimize(alg, T, model, q0; show_progress=PROGRESS)

        Δλ0 = sum(abs2, q0.location - μ_true) + sum(abs2, q0.scale - L_true)
        Δλ = sum(abs2, q_out.location - μ_true) + sum(abs2, q_out.scale - L_true)

        @test Δλ ≤ Δλ0/2
    end

    @testset "subsampling" begin
        n_data = 8

        @testset "estimate_objective batchsize=$(batchsize)" for batchsize in [1, 3, 4]
            modelstats = subsamplednormal(Random.default_rng(), n_data)
            (; model, n_dims, μ_true, L_true) = modelstats

            L0 = LowerTriangular(Matrix{Float64}(I, n_dims, n_dims))
            q0 = FullRankGaussian(zeros(Float64, n_dims), L0)
            operator = ClipScale()

            subsampling = ReshufflingBatchSubsampling(1:n_data, batchsize)
            alg = KLMinScoreGradDescent(AD; n_samples=10, operator)
            alg_sub = KLMinScoreGradDescent(AD; n_samples=10, subsampling, operator)

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
            alg_sub = KLMinScoreGradDescent(
                AD; n_samples=10, subsampling, operator=ClipScale()
            )

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

        @testset "convergence" begin
            modelstats = subsamplednormal(Random.default_rng(), n_data)
            (; model, n_dims, μ_true, L_true) = modelstats

            L0 = LowerTriangular(Matrix{Float64}(I, n_dims, n_dims))
            q0 = FullRankGaussian(zeros(Float64, n_dims), L0)

            T = 1000
            batchsize = 1
            optimizer = Descent(1e-3)
            subsampling = ReshufflingBatchSubsampling(1:n_data, batchsize)
            alg_sub = KLMinScoreGradDescent(
                AD; n_samples=100, optimizer, subsampling, operator=ClipScale()
            )

            q, stats, _ = optimize(alg_sub, T, model, q0; show_progress=PROGRESS)

            Δλ0 = sum(abs2, q0.location - μ_true) + sum(abs2, q0.scale - L_true)
            Δλ = sum(abs2, q.location - μ_true) + sum(abs2, q.scale - L_true)

            @test Δλ ≤ Δλ0/2
        end
    end
end
