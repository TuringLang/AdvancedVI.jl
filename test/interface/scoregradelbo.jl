
using Test

@testset "interface ScoreGradELBO" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    modelstats = normal_meanfield(rng, Float64)

    @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

    q0 = TuringDiagMvNormal(zeros(Float64, n_dims), ones(Float64, n_dims))

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

    @testset "baseline_window" begin
        T = 100
        adtype = AutoForwardDiff()

        obj = ScoreGradELBO(10)
        _, _, stats, _ = optimize(rng, model, obj, q0, T; show_progress=false, adtype)
        @test isfinite(last(stats).elbo)

        obj = ScoreGradELBO(10; baseline_window_size=0)
        _, _, stats, _ = optimize(rng, model, obj, q0, T; show_progress=false, adtype)
        @test isfinite(last(stats).elbo)

        obj = ScoreGradELBO(10; baseline_window_size=1)
        _, _, stats, _ = optimize(rng, model, obj, q0, T; show_progress=false, adtype)
        @test isfinite(last(stats).elbo)
    end
end
