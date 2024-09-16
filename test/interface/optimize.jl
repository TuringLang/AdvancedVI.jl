
using Test

@testset "interface optimize" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    T = 1000
    modelstats = normal_meanfield(rng, Float64)

    @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

    # Global Test Configurations
    q0 = TuringDiagMvNormal(zeros(Float64, n_dims), ones(Float64, n_dims))
    obj = RepGradELBO(10)

    adtype = AutoForwardDiff()
    optimizer = Optimisers.Adam(1e-2)
    averager = PolynomialAveraging()

    @testset "default_rng" begin
        optimize(model, obj, q0, T; optimizer, averager, show_progress=false, adtype)
    end

    @testset "callback" begin
        rng = StableRNG(seed)
        test_values = rand(rng, T)

        callback(; stat, args...) = (test_value=test_values[stat.iteration],)

        rng = StableRNG(seed)
        _, _, stats, _ = optimize(
            rng, model, obj, q0, T; show_progress=false, adtype, callback
        )
        @test [stat.test_value for stat in stats] == test_values
    end

    rng = StableRNG(seed)
    q_avg_ref, q_ref, _, _ = optimize(
        rng, model, obj, q0, T; optimizer, averager, show_progress=false, adtype
    )

    @testset "warm start" begin
        rng = StableRNG(seed)

        T_first = div(T, 2)
        T_last = T - T_first

        _, q_first, _, state = optimize(
            rng, model, obj, q0, T_first; optimizer, averager, show_progress=false, adtype
        )

        q_avg, q, _, _ = optimize(
            rng,
            model,
            obj,
            q_first,
            T_last;
            optimizer,
            averager,
            show_progress=false,
            state_init=state,
            adtype,
        )

        @test q == q_ref
        @test q_avg == q_avg_ref
    end
end
