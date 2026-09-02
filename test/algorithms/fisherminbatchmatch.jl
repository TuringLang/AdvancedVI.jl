
struct FixedBatchRNG{A<:AbstractMatrix} <: AbstractRNG
    values::A
end

function Random.randn!(rng::FixedBatchRNG, x::AbstractMatrix)
    size(x) == size(rng.values) || throw(DimensionMismatch())
    return copyto!(x, rng.values)
end

@testset "FisherMinBatchMatch" begin
    begin
        modelstats = normal_meanfield(Random.default_rng(), Float64; capability=2)
        (; model, n_dims, μ_true, L_true) = modelstats

        alg = FisherMinBatchMatch()
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

            obj_est = estimate_objective(alg, q_true, model; n_samples=10^6)
            @test obj_est ≈ 0 atol = 1e-2
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

        @testset "population batch moments" begin
            target_mean = [0.5]
            target_var = Diagonal([0.7])
            model = TestNormal(
                target_mean, target_var, LogDensityProblems.LogDensityOrder{1}()
            )
            q0 = FullRankGaussian([0.2], LowerTriangular(reshape([1.3], 1, 1)))
            u = reshape([-1.0, 0.25, 2.0], 1, :)
            alg = FisherMinBatchMatch(; n_samples=size(u, 2))
            rng = FixedBatchRNG(u)
            state = AdvancedVI.init(rng, alg, q0, model)

            state′, _, _ = AdvancedVI.step(rng, alg, state, nothing)

            z = q0.scale * u .+ q0.location
            grad = -(z .- target_mean) ./ target_var[1, 1]
            zbar, C = mean_and_cov(z, 2; corrected=false)
            gbar, Γ = mean_and_cov(grad, 2; corrected=false)
            λ = size(u, 2)
            μmz = q0.location - zbar
            U = Symmetric(λ * Γ + (λ / (1 + λ) * gbar) * gbar')
            V = Symmetric(cov(q0) + λ * C + (λ / (1 + λ) * μmz) * μmz')
            Σ_expected = Hermitian(2 * V / (I + real(sqrt(I + 4 * U * V))))

            _, C_corrected = mean_and_cov(z, 2; corrected=true)
            _, Γ_corrected = mean_and_cov(grad, 2; corrected=true)
            U_corrected = Symmetric(λ * Γ_corrected + (λ / (1 + λ) * gbar) * gbar')
            V_corrected = Symmetric(cov(q0) + λ * C_corrected + (λ / (1 + λ) * μmz) * μmz')
            Σ_corrected = Hermitian(
                2 * V_corrected / (I + real(sqrt(I + 4 * U_corrected * V_corrected)))
            )

            @test state′.sigma ≈ Σ_expected
            @test !isapprox(state′.sigma, Σ_corrected)
        end

        @testset "one-sample batch" begin
            model = TestNormal(
                [0.5], Diagonal([0.7]), LogDensityProblems.LogDensityOrder{1}()
            )
            q0 = FullRankGaussian([0.2], LowerTriangular(reshape([1.3], 1, 1)))
            rng = FixedBatchRNG(reshape([0.25], 1, 1))
            alg = FisherMinBatchMatch(; n_samples=1)
            state = AdvancedVI.init(rng, alg, q0, model)

            state′, _, _ = AdvancedVI.step(rng, alg, state, nothing)

            @test all(isfinite, state′.sigma)
            @test all(isfinite, state′.q.location)
        end
    end

    @testset "error low capability" begin
        modelstats = normal_meanfield(Random.default_rng(), Float64; capability=0)
        (; model, n_dims) = modelstats

        alg = FisherMinBatchMatch()

        L0 = LowerTriangular(Matrix{Float64}(I, n_dims, n_dims))
        q0 = FullRankGaussian(zeros(Float64, n_dims), L0)
        @test_throws "first-order" optimize(alg, 1, model, q0)
    end

    @testset "type stability type=$(realtype), capability=$(capability)" for realtype in [
            Float64, Float32
        ],
        capability in [1, 2]

        modelstats = normal_meanfield(Random.default_rng(), realtype; capability)
        (; model, μ_true, L_true, n_dims, strong_convexity, is_meanfield) = modelstats

        alg = FisherMinBatchMatch()
        T = 10

        L0 = LowerTriangular(Matrix{realtype}(I, n_dims, n_dims))
        q0 = FullRankGaussian(zeros(realtype, n_dims), L0)

        q, _, _ = optimize(alg, T, model, q0; show_progress=PROGRESS)

        @test eltype(q.location) == eltype(μ_true)
        @test eltype(q.scale) == eltype(L_true)
    end

    @testset "convergence" begin
        modelstats = normal_meanfield(Random.default_rng(), Float64)
        (; model, μ_true, L_true, n_dims, strong_convexity, is_meanfield) = modelstats

        T = 1000
        alg = FisherMinBatchMatch()

        q_avg, _, _ = optimize(alg, T, model, q0; show_progress=PROGRESS)

        Δλ0 = sum(abs2, q0.location - μ_true) + sum(abs2, q0.scale - L_true)
        Δλ = sum(abs2, q_avg.location - μ_true) + sum(abs2, q_avg.scale - L_true)

        @test Δλ ≤ Δλ0 / 2
    end

    @testset "subsampling" begin
        n_data = 8

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
            alg_sub = FisherMinBatchMatch(; subsampling)

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
            subsampling = ReshufflingBatchSubsampling(1:n_data, batchsize)
            alg_sub = FisherMinBatchMatch(; subsampling)

            q, stats, _ = optimize(alg_sub, T, model, q0; show_progress=PROGRESS)

            Δλ0 = sum(abs2, q0.location - μ_true) + sum(abs2, q0.scale - L_true)
            Δλ = sum(abs2, q.location - μ_true) + sum(abs2, q.scale - L_true)

            @test Δλ ≤ Δλ0 / 2
        end
    end
end
