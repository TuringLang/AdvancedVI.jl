AD_scoregradelbo_locationscale_bijectors = if TEST_GROUP == "Enzyme"
    Dict(
        :Enzyme => AutoEnzyme(;
            mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation=Enzyme.Const,
        ),
    )
else
    Dict(
        :ForwarDiff => AutoForwardDiff(),
        :ReverseDiff => AutoReverseDiff(),
        #:Zygote => AutoZygote(),
        #:Mooncake => AutoMooncake(; config=Mooncake.Config()),
    )
end

@testset "inference ScoreGradELBO VILocationScale Bijectors" begin
    @testset "$(modelname) $(realtype) $(adbackname)" for realtype in [Float64, Float32],
        (modelname, modelconstr) in
        Dict(:NormalLogNormalMeanField => normallognormal_meanfield),
        (adbackname, adtype) in AD_scoregradelbo_locationscale_bijectors

        seed = (0x38bef07cf9cc549d)
        rng = StableRNG(seed)

        modelstats = modelconstr(rng, realtype)
        (; model, μ_true, L_true, n_dims, strong_convexity, is_meanfield) = modelstats

        T = 1000
        η = 1e-4
        opt = Optimisers.Descent(η)
        alg = KLMinScoreGradDescent(adtype; n_samples=10, optimizer=opt)

        b = Bijectors.bijector(model)
        b⁻¹ = inverse(b)
        μ0 = Zeros(realtype, n_dims)
        L0 = Diagonal(Ones(realtype, n_dims))

        q0_η = if is_meanfield
            MeanFieldGaussian(zeros(realtype, n_dims), Diagonal(ones(realtype, n_dims)))
        else
            L0 = LowerTriangular(Matrix{realtype}(I, n_dims, n_dims))
            FullRankGaussian(zeros(realtype, n_dims), L0)
        end
        q0_z = Bijectors.transformed(q0_η, b⁻¹)

        # For small enough η, the error of SGD, Δλ, is bounded as
        #     Δλ ≤ ρ^T Δλ0 + O(η),
        # where ρ = 1 - ημ, μ is the strong convexity constant.
        contraction_rate = 1 - η * strong_convexity

        @testset "convergence" begin
            Δλ0 = sum(abs2, μ0 - μ_true) + sum(abs2, L0 - L_true)
            q_avg, stats, _ = optimize(rng, alg, T, model, q0_z; show_progress=PROGRESS)

            μ = q_avg.dist.location
            L = q_avg.dist.scale
            Δλ = sum(abs2, μ - μ_true) + sum(abs2, L - L_true)

            @test Δλ ≤ contraction_rate^(T / 2) * Δλ0
            @test eltype(μ) == eltype(μ_true)
            @test eltype(L) == eltype(L_true)
        end

        @testset "determinism" begin
            rng = StableRNG(seed)
            q_avg, stats, _ = optimize(rng, alg, T, model, q0_z; show_progress=PROGRESS)
            μ = q_avg.dist.location
            L = q_avg.dist.scale

            rng_repl = StableRNG(seed)
            q_avg, stats, _ = optimize(
                rng_repl, alg, T, model, q0_z; show_progress=PROGRESS
            )
            μ_repl = q_avg.dist.location
            L_repl = q_avg.dist.scale
            @test μ ≈ μ_repl rtol = 1e-3
            @test L ≈ L_repl rtol = 1e-3
        end
    end
end
