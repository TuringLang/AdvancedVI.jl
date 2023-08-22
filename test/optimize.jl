
using ReTest

@testset "optimize" begin
    seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
    rng  = Philox4x(UInt64, seed, 8)

    T = 1000
    modelstats = normallognormal_meanfield(Float64; rng)

    @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

    # Global Test Configurations
    b⁻¹ = Bijectors.bijector(model) |> inverse
    μ₀  = zeros(Float64, n_dims)
    L₀  = ones(Float64, n_dims) |> Diagonal
    q₀  = VIMeanFieldGaussian(μ₀, L₀)
    obj = ADVI(model, 10; invbij=b⁻¹)

    adbackend = AutoForwardDiff()
    optimizer = Optimisers.Adam(1e-2)

    rng                 = Philox4x(UInt64, seed, 8)
    q_ref, stats_ref, _ = optimize(
        obj, q₀, T;
        optimizer,
        show_progress = false,
        rng,
        adbackend,
    )
    λ_ref, _ = Optimisers.destructure(q_ref)

    @testset "restructure" begin
        λ₀, re  = Optimisers.destructure(q₀)

        rng         = Philox4x(UInt64, seed, 8)
        λ, stats, _ = optimize(
            obj, re, λ₀, T;
            optimizer,
            show_progress = false,
            rng,
            adbackend,
        )
        @test λ     == λ_ref
        @test stats == stats_ref
    end

    @testset "callback" begin
        rng = Philox4x(UInt64, seed, 8)
        test_values = rand(rng, T)

        callback!(; stat, est_state, restructure, λ, g) = begin
            (test_value = test_values[stat.iteration],)
        end

        rng         = Philox4x(UInt64, seed, 8)
        _, stats, _ = optimize(
            obj, q₀, T;
            show_progress = false,
            rng,
            adbackend,
            callback!
        )
        @test [stat.test_value for stat ∈ stats] == test_values
    end
end
