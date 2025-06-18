
AD_scoregradelbo_interface = if TEST_GROUP == "Enzyme"
    [AutoEnzyme()]
else
    [
        AutoForwardDiff(),
        AutoReverseDiff(),
        AutoZygote(),
        AutoMooncake(; config=Mooncake.Config()),
    ]
end

@testset "interface ScoreGradELBO" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    modelstats = normal_meanfield(rng, Float64)

    (; model, μ_true, L_true, n_dims, is_meanfield) = modelstats

    q0 = MeanFieldGaussian(zeros(n_dims), Diagonal(ones(n_dims)))

    @testset "basic" begin
        @testset for adtype in AD_scoregradelbo_interface, n_montecarlo in [1, 10]
            alg = ParamSpaceSGD(
                model,
                ScoreGradELBO(n_montecarlo),
                adtype,
                Descent(1e-5),
                PolynomialAveraging(),
                IdentityOperator()
            )
            _, info, _ = optimize(rng, alg, 10, q0; show_progress=false)
            @assert isfinite(last(info).elbo)
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
