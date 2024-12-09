
AD_repgradelbo_interface = if TEST_GROUP == "Enzyme"
    [AutoEnzyme()]
else
    [
        AutoForwardDiff(),
        AutoReverseDiff(),
        AutoZygote(),
        AutoMooncake(; config=Mooncake.Config()),
    ]
end

@testset "interface RepGradELBO" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    modelstats = normal_meanfield(rng, Float64)

    (; model, μ_true, L_true, n_dims, is_meanfield) = modelstats

    q0 = MeanFieldGaussian(zeros(n_dims), Diagonal(ones(n_dims)))

    @testset "basic" begin
        @testset for adtype in AD_repgradelbo_interface, n_montecarlo in [1, 10]
            obj = RepGradELBO(n_montecarlo)
            _, _, stats, _ = optimize(
                rng,
                model,
                obj,
                q0,
                10;
                optimizer=Descent(1e-5),
                show_progress=false,
                adtype=adtype,
            )
            @assert isfinite(last(stats).elbo)
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

    @testset for adtype in AD_repgradelbo_interface, n_montecarlo in [1, 10]
        q_true = MeanFieldGaussian(
            Vector{eltype(μ_true)}(μ_true), Diagonal(Vector{eltype(L_true)}(diag(L_true)))
        )
        params, re = Optimisers.destructure(q_true)
        obj = RepGradELBO(n_montecarlo; entropy=StickingTheLandingEntropy())
        out = DiffResults.DiffResult(zero(eltype(params)), similar(params))

        aux = (
            rng=rng, obj=obj, problem=model, restructure=re, q_stop=q_true, adtype=adtype
        )
        AdvancedVI.value_and_gradient!(
            adtype, AdvancedVI.estimate_repgradelbo_ad_forward, params, aux, out
        )
        grad = DiffResults.gradient(out)
        @test norm(grad) ≈ 0 atol = 1e-5
    end
end
