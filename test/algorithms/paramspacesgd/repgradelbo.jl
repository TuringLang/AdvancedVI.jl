
@testset "interface RepGradELBO" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    modelstats = normal_meanfield(rng, Float64)

    (; model, μ_true, L_true, n_dims, is_meanfield) = modelstats

    q0 = MeanFieldGaussian(zeros(n_dims), Diagonal(ones(n_dims)))

    @testset "basic" begin
        @testset for n_montecarlo in [1, 10]
            alg = KLMinRepGradDescent(
                AD;
                n_samples=n_montecarlo,
                operator=IdentityOperator(),
                averager=PolynomialAveraging(),
            )
            _, info, _ = optimize(rng, alg, 10, model, q0; show_progress=false)
            @test isfinite(last(info).elbo)
        end
    end

    @testset "without mixed ad" begin
        @testset for n_montecarlo in [1, 10]
            alg = KLMinRepGradDescent(
                AD;
                n_samples=n_montecarlo,
                operator=IdentityOperator(),
                averager=PolynomialAveraging(),
            )
            _, info, _ = optimize(rng, alg, 10, model, q0; show_progress=false)
            @test isfinite(last(info).elbo)
        end
    end

    obj = RepGradELBO(10)
    rng = StableRNG(seed)
    elbo_ref = estimate_objective(rng, obj, q0, model; n_samples=10^4)

    @testset "determinism" begin
        rng = StableRNG(seed)
        elbo = estimate_objective(rng, obj, q0, model; n_samples=10^4)
        @test elbo == elbo_ref
    end

    @testset "default_rng" begin
        elbo = estimate_objective(obj, q0, model; n_samples=10^4)
        @test elbo ≈ elbo_ref rtol = 0.1
    end
end

@testset "interface RepGradELBO STL variance reduction" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    modelstats = normal_meanfield(rng, Float64)
    (; model, μ_true, L_true, n_dims, is_meanfield) = modelstats

    model = ADgradient(AutoForwardDiff(), model)
    mixed_ad = AdvancedVI.MixedADLogDensityProblem(model)

    @testset for n_montecarlo in [1, 10]
        q_true = MeanFieldGaussian(
            Vector{eltype(μ_true)}(μ_true), Diagonal(Vector{eltype(L_true)}(diag(L_true)))
        )
        params, re = Optimisers.destructure(q_true)
        obj = RepGradELBO(n_montecarlo; entropy=StickingTheLandingEntropy())
        out = DiffResults.DiffResult(zero(eltype(params)), similar(params))

        aux = (rng=rng, obj=obj, problem=mixed_ad, restructure=re, q_stop=q_true, adtype=AD)
        AdvancedVI._value_and_gradient!(
            AdvancedVI.estimate_repgradelbo_ad_forward, out, AD, params, aux
        )
        grad = DiffResults.gradient(out)
        @test norm(grad) ≈ 0 atol = 1e-5
    end
end
