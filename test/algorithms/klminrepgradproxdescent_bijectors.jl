
@testset "KLMinRepGradProxDescent with Bijectors" begin
    begin
        modelstats = normallognormal_meanfield(Random.default_rng(), Float64)
        (; model, n_dims, μ_true, L_true) = modelstats

        b = Bijectors.bijector(model)
        binv = inverse(b)

        q0_unconstr = MeanFieldGaussian(zeros(n_dims), Diagonal(ones(n_dims)))
        q0 = Bijectors.transformed(q0_unconstr, binv)

        @testset "estimate_objective" begin
            alg = KLMinRepGradProxDescent(AD)
            q_true_unconstr = MeanFieldGaussian(Vector(μ_true), Diagonal(L_true))
            q_true = Bijectors.transformed(q_true_unconstr, binv)

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
            alg = KLMinRepGradProxDescent(AD)

            seed = (0x38bef07cf9cc549d)
            rng = StableRNG(seed)
            T = 10

            q_out, _, _ = optimize(rng, alg, T, model, q0; show_progress=PROGRESS)
            μ = q_out.dist.location
            L = q_out.dist.scale

            rng_repl = StableRNG(seed)
            q_out, _, _ = optimize(rng_repl, alg, T, model, q0; show_progress=PROGRESS)
            μ_repl = q_out.dist.location
            L_repl = q_out.dist.scale
            @test μ == μ_repl
            @test L == L_repl
        end
    end

    @testset "type stability realtype=$(realtype)" for realtype in [Float32, Float64]
        modelstats = normallognormal_meanfield(Random.default_rng(), realtype)
        (; model, n_dims, μ_true, L_true) = modelstats

        T = 1
        alg = KLMinRepGradProxDescent(AD; n_samples=10)
        q0_unconstr = MeanFieldGaussian(
            zeros(realtype, n_dims), Diagonal(ones(realtype, n_dims))
        )
        q0 = Bijectors.transformed(q0_unconstr, binv)

        q_out, info, _ = optimize(alg, T, model, q0; show_progress=PROGRESS)

        @test eltype(q_out.dist.location) == realtype
        @test eltype(q_out.dist.scale) == realtype
        @test typeof(first(info).elbo) == realtype
    end

    @testset "convergence $(entropy)" for entropy_zerograd in [
        ClosedFormEntropyZeroGradient(), StickingTheLandingEntropyZeroGradient()
    ]
        modelstats = normallognormal_meanfield(Random.default_rng(), Float64)
        (; model, μ_true, L_true, is_meanfield) = modelstats

        T = 1000
        optimizer = Descent(1e-3)
        alg = KLMinRepGradProxDescent(AD; n_samples=10, optimizer, entropy_zerograd)

        q0_unconstr = MeanFieldGaussian(zeros(n_dims), Diagonal(ones(n_dims)))
        q0 = Bijectors.transformed(q0_unconstr, binv)

        q_out, _, _ = optimize(alg, T, model, q0; show_progress=PROGRESS)

        Δλ0 = sum(abs2, q0.dist.location - μ_true) + sum(abs2, q0.dist.scale - L_true)
        Δλ = sum(abs2, q_out.dist.location - μ_true) + sum(abs2, q_out.dist.scale - L_true)

        @test Δλ ≤ Δλ0/2
    end
end
