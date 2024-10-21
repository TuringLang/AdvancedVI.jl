
using Test

@testset "interface RepGradELBO" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    modelstats = normal_meanfield(rng, Float64)

    (; model, μ_true, L_true, n_dims, is_meanfield) = modelstats

    q0 = TuringDiagMvNormal(zeros(Float64, n_dims), ones(Float64, n_dims))

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

AD_repgradelbo_stl = if AD_GROUP == "General"
    Dict(
        :ForwarDiff => AutoForwardDiff(),
        :ReverseDiff => AutoReverseDiff(),
        :Zygote => AutoZygote(),
        :Mooncake => AutoMooncake(; config=Mooncake.Config()),
    )
elseif AD_GROUP == "Enzyme"
    Dict(:Enzyme => AutoEnzyme())
end

@testset "interface RepGradELBO STL variance reduction" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    modelstats = normal_meanfield(rng, Float64)
    (; model, μ_true, L_true, n_dims, is_meanfield) = modelstats

    @testset for adtype in AD_repgradelbo_stl
        q_true = MeanFieldGaussian(
            Vector{eltype(μ_true)}(μ_true), Diagonal(Vector{eltype(L_true)}(diag(L_true)))
        )
        params, re = Optimisers.destructure(q_true)
        obj = RepGradELBO(10; entropy=StickingTheLandingEntropy())
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
