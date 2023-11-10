
using Test

@testset "advi" begin
    seed = (0x38bef07cf9cc549d)
    rng  = StableRNG(seed)

    @testset "with bijector"  begin
        modelstats = normallognormal_meanfield(rng, Float64)

        @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

        b⁻¹  = Bijectors.bijector(model) |> inverse
        q₀_η = TuringDiagMvNormal(zeros(Float64, n_dims), ones(Float64, n_dims))
        q₀_z = Bijectors.transformed(q₀_η, b⁻¹)
        obj  = ADVI(10)

        rng      = StableRNG(seed)
        elbo_ref = estimate_objective(rng, obj, q₀_z, model; n_samples=10^4)

        @testset "determinism" begin
            rng  = StableRNG(seed)
            elbo = estimate_objective(rng, obj, q₀_z, model; n_samples=10^4)
            @test elbo == elbo_ref
        end

        @testset "default_rng" begin
            elbo = estimate_objective(obj, q₀_z, model; n_samples=10^4)
            @test elbo ≈ elbo_ref rtol=0.1
        end
    end

    @testset "without bijector"  begin
        modelstats = normal_meanfield(rng, Float64)

        @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

        q₀_z = TuringDiagMvNormal(zeros(Float64, n_dims), ones(Float64, n_dims))

        obj      = ADVI(10)
        rng      = StableRNG(seed)
        elbo_ref = estimate_objective(rng, obj, q₀_z, model; n_samples=10^4)

        @testset "determinism" begin
            rng  = StableRNG(seed)
            elbo = estimate_objective(rng, obj, q₀_z, model; n_samples=10^4)
            @test elbo == elbo_ref
        end

        @testset "default_rng" begin
            elbo = estimate_objective(obj, q₀_z, model; n_samples=10^4)
            @test elbo ≈ elbo_ref rtol=0.1
        end
    end
end
