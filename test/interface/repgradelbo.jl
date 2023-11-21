
using Test

@testset "interface RepGradELBO" begin
    seed = (0x38bef07cf9cc549d)
    rng  = StableRNG(seed)

    modelstats = normal_meanfield(rng, Float64)

    @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

    q0 = TuringDiagMvNormal(zeros(Float64, n_dims), ones(Float64, n_dims))

    obj      = RepGradELBO(10)
    rng      = StableRNG(seed)
    elbo_ref = estimate_objective(rng, obj, q0, model; n_samples=10^4)

    @testset "determinism" begin
        rng  = StableRNG(seed)
        elbo = estimate_objective(rng, obj, q0, model; n_samples=10^4)
        @test elbo == elbo_ref
    end

    @testset "default_rng" begin
        elbo = estimate_objective(obj, q0, model; n_samples=10^4)
        @test elbo ≈ elbo_ref rtol=0.1
    end
end
