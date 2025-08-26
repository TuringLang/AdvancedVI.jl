@testset "interface ScoreGradELBO" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    modelstats = normal_meanfield(rng, Float64)

    (; model, μ_true, L_true, n_dims, is_meanfield) = modelstats

    q0 = MeanFieldGaussian(zeros(n_dims), Diagonal(ones(n_dims)))
    q0_trans = Bijectors.transformed(q0, identity)

    @testset "basic" begin
        @testset for n_montecarlo in [1, 10]
            alg = KLMinScoreGradDescent(
                AD; n_samples=n_montecarlo, operator=ClipScale(), optimizer=Descent(1e-5)
            )

            _, info, _ = optimize(rng, alg, 10, model, q0; show_progress=false)
            @assert isfinite(last(info).elbo)

            _, info, _ = optimize(rng, alg, 10, model, q0_trans; show_progress=false)
            @assert isfinite(last(info).elbo)
        end
    end

    @testset "warn MvLocationScale with IdentityOperator" begin
        @test_nowarn begin
            alg = KLMinScoreGradDescent(AD; operator=ClipScale())
            optimize(rng, alg, 1, model, q0; show_progress=false)
        end
        @test_warn "IdentityOperator" begin
            alg = KLMinScoreGradDescent(AD; operator=IdentityOperator())
            optimize(rng, alg, 1, model, q0; show_progress=false)
        end
        @test_nowarn begin
            alg = KLMinScoreGradDescent(AD; operator=ClipScale())
            optimize(rng, alg, 1, model, q0_trans; show_progress=false)
        end
        @test_warn "IdentityOperator" begin
            alg = KLMinScoreGradDescent(AD; operator=IdentityOperator())
            optimize(rng, alg, 1, model, q0_trans; show_progress=false)
        end
    end

    obj = ScoreGradELBO(10)
    rng = StableRNG(seed)
    elbo_ref = estimate_objective(rng, obj, q0, model; n_samples=10^4)

    @testset "determinism" begin
        rng = StableRNG(seed)
        elbo = estimate_objective(rng, obj, q0, model; n_samples=10^4)
        @test elbo == elbo_ref
    end

    @testset "default_rng" begin
        elbo = estimate_objective(obj, q0, model; n_samples=10^4)
        @test elbo ≈ elbo_ref rtol = 0.2
    end
end
