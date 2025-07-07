AD_repgradelbo_locationscale = if TEST_GROUP == "Enzyme"
    Dict(
        :Enzyme => AutoEnzyme(;
            mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation=Enzyme.Const,
        ),
    )
else
    Dict(
        :ReverseDiff => AutoReverseDiff(),
        :Zygote => AutoZygote(),
        :Mooncake => AutoMooncake(; config=Mooncake.Config()),
    )
end

@testset "inference RepGradELBO VILocationScale" begin
    @testset "$(modelname) $(objname) $(realtype) $(adbackname)" for realtype in
                                                                     [Float64, Float32],
        (modelname, modelconstr) in
        Dict(:Normal => normal_meanfield, :Normal => normal_fullrank),
        (objname, objective) in Dict(
            :RepGradELBOClosedFormEntropy => RepGradELBO(10),
            :RepGradELBOStickingTheLanding =>
                RepGradELBO(10; entropy=StickingTheLandingEntropy()),
        ),
        (adbackname, adtype) in AD_repgradelbo_locationscale

        seed = (0x38bef07cf9cc549d)
        rng = StableRNG(seed)

        modelstats = modelconstr(rng, realtype)
        (; model, μ_true, L_true, n_dims, strong_convexity, is_meanfield) = modelstats

        model_ad = ADgradient(AutoForwardDiff(), model)

        T = 1000
        η = 1e-3
        alg = BBVIRepGrad(adtype; optimizer=Descent(η))

        q0 = if is_meanfield
            MeanFieldGaussian(zeros(realtype, n_dims), Diagonal(ones(realtype, n_dims)))
        else
            L0 = LowerTriangular(Matrix{realtype}(I, n_dims, n_dims))
            FullRankGaussian(zeros(realtype, n_dims), L0)
        end

        # For small enough η, the error of SGD, Δλ, is bounded as
        #     Δλ ≤ ρ^T Δλ0 + O(η),
        # where ρ = 1 - ημ, μ is the strong convexity constant.
        contraction_rate = 1 - η * strong_convexity

        @testset "convergence" begin
            Δλ0 = sum(abs2, q0.location - μ_true) + sum(abs2, q0.scale - L_true)
            q_avg, stats, _ = optimize(rng, alg, T, model_ad, q0; show_progress=PROGRESS)

            μ = q_avg.location
            L = q_avg.scale
            Δλ = sum(abs2, μ - μ_true) + sum(abs2, L - L_true)

            @test Δλ ≤ contraction_rate^(T / 2) * Δλ0
            @test eltype(μ) == eltype(μ_true)
            @test eltype(L) == eltype(L_true)
        end

        @testset "determinism" begin
            rng = StableRNG(seed)
            q_avg, stats, _ = optimize(rng, alg, T, model_ad, q0; show_progress=PROGRESS)
            μ = q_avg.location
            L = q_avg.scale

            rng_repl = StableRNG(seed)
            q_avg, stats, _ = optimize(
                rng_repl, alg, T, model_ad, q0; show_progress=PROGRESS
            )
            μ_repl = q_avg.location
            L_repl = q_avg.scale
            @test μ == μ_repl
            @test L == L_repl
        end
    end
end
