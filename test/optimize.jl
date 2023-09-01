
using Test

@testset "optimize" begin
    seed = (0x38bef07cf9cc549d)
    rng  = StableRNG(seed)

    T = 1000
    modelstats = normallognormal_meanfield(Float64; rng)

    @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

    # Global Test Configurations
    b⁻¹  = Bijectors.bijector(model) |> inverse
    q₀_η = TuringDiagMvNormal(zeros(Float64, n_dims), ones(Float64, n_dims))
    q₀_z = Bijectors.transformed(q₀_η, b⁻¹)
    obj  = ADVI(10)

    adbackend = AutoForwardDiff()
    optimizer = Optimisers.Adam(1e-2)

    rng  = StableRNG(seed)
    q_ref, stats_ref, _ = optimize(
        model, obj, q₀_z, T;
        optimizer,
        show_progress = false,
        rng,
        adbackend,
    )
    λ_ref, _ = Optimisers.destructure(q_ref)

    @testset "restructure" begin
        λ₀, re  = Optimisers.destructure(q₀_z)

        rng  = StableRNG(seed)
        λ, stats, _ = optimize(
            model, obj, re, λ₀, T;
            optimizer,
            show_progress = false,
            rng,
            adbackend,
        )
        @test λ     == λ_ref
        @test stats == stats_ref
    end

    @testset "callback" begin
        rng  = StableRNG(seed)
        test_values = rand(rng, T)

        callback!(; stat, restructure, λ, g) = begin
            (test_value = test_values[stat.iteration],)
        end

        rng  = StableRNG(seed)
        _, stats, _ = optimize(
            model, obj, q₀_z, T;
            show_progress = false,
            rng,
            adbackend,
            callback!
        )
        @test [stat.test_value for stat ∈ stats] == test_values
    end

    @testset "warm start" begin
        rng  = StableRNG(seed)

        T_first = div(T,2)
        T_last  = T - T_first

        q_first, _, state = optimize(
            model, obj, q₀_z, T_first;
            optimizer,
            show_progress = false,
            rng,
            adbackend
        )

        q, stats, _ = optimize(
            model, obj, q_first, T_last;
            optimizer,
            show_progress = false,
            state,
            rng,
            adbackend
        )
        @test q == q_ref
    end
end
