
@testset "interface optimize" begin
    seed = (0x38bef07cf9cc549d)
    rng = StableRNG(seed)

    T = 1000
    modelstats = normal_meanfield(rng, Float64)

    (; model, μ_true, L_true, n_dims, is_meanfield) = modelstats

    q0 = MeanFieldGaussian(zeros(Float64, n_dims), Diagonal(ones(Float64, n_dims)))
    obj = RepGradELBO(10)

    adtype = AutoForwardDiff()
    optimizer = Optimisers.Adam(1e-2)
    averager = PolynomialAveraging()

    alg = ParamSpaceSGD(obj, adtype, optimizer, averager, IdentityOperator())

    @testset "default_rng" begin
        optimize(alg, T, model, q0; show_progress=false)
    end

    @testset "callback" begin
        test_values = rand(rng, T)

        callback(; iteration, args...) = (test_value=test_values[iteration],)

        _, info, _ = optimize(alg, T, model, q0; show_progress=false, callback)
        @test [i.test_value for i in info] == test_values
    end

    rng = StableRNG(seed)
    q_avg_ref, _, _ = optimize(rng, alg, T, model, q0; show_progress=false)

    @testset "warm start" begin
        rng = StableRNG(seed)

        T_first = div(T, 2)
        T_last = T - T_first

        _, _, state = optimize(rng, alg, T_first, model, q0; show_progress=false)
        q_avg, _, _ = optimize(rng, alg, T_last, model, q0; show_progress=false, state)

        @test q_avg == q_avg_ref
    end
end
