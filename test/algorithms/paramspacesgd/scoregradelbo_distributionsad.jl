AD_scoregradelbo_distributionsad = if TEST_GROUP == "Enzyme"
    Dict(
        :Enzyme => AutoEnzyme(;
            mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation=Enzyme.Const,
        ),
    )
else
    Dict(
        :ForwarDiff => AutoForwardDiff(),
        #:ReverseDiff => AutoReverseDiff(),
        :Zygote => AutoZygote(),
        #:Mooncake => AutoMooncake(; config=Mooncake.Config()),
    )
end

@testset "inference ScoreGradELBO DistributionsAD" begin
    @testset "$(modelname) $(objname) $(realtype) $(adbackname)" for realtype in
                                                                     [Float64, Float32],
        (modelname, modelconstr) in Dict(:Normal => normal_meanfield),
        (objname, objective) in Dict(:ScoreGradELBO => ScoreGradELBO(10)),
        (adbackname, adtype) in AD_scoregradelbo_distributionsad

        seed = (0x38bef07cf9cc549d)
        rng = StableRNG(seed)

        modelstats = modelconstr(rng, realtype)
        (; model, μ_true, L_true, n_dims, strong_convexity, is_meanfield) = modelstats

        T = 1000
        η = 1e-4
        opt = Optimisers.Descent(η)
        op = IdentityOperator()
        avg = PolynomialAveraging()
        alg = ParamSpaceSGD(model, objective, adtype, opt, avg, op)

        # For small enough η, the error of SGD, Δλ, is bounded as
        #     Δλ ≤ ρ^T Δλ0 + O(η),
        # where ρ = 1 - ημ, μ is the strong convexity constant.
        contraction_rate = 1 - η * strong_convexity

        μ0 = zeros(realtype, n_dims)
        L0 = Diagonal(ones(realtype, n_dims))
        q0 = TuringDiagMvNormal(μ0, diag(L0))

        @testset "convergence" begin
            Δλ0 = sum(abs2, μ0 - μ_true) + sum(abs2, L0 - L_true)
            q_avg, stats, _ = optimize(rng, alg, T, q0; show_progress=PROGRESS)

            μ = mean(q_avg)
            L = sqrt(cov(q_avg))
            Δλ = sum(abs2, μ - μ_true) + sum(abs2, L - L_true)

            @test Δλ ≤ contraction_rate^(T / 2) * Δλ0
            @test eltype(μ) == eltype(μ_true)
            @test eltype(L) == eltype(L_true)
        end

        @testset "determinism" begin
            rng = StableRNG(seed)
            q_avg, stats, _ = optimize(rng, alg, T, q0; show_progress=PROGRESS)
            μ = mean(q_avg)
            L = sqrt(cov(q_avg))

            rng_repl = StableRNG(seed)
            q_avg, stats, _ = optimize(rng_repl, alg, T, q0; show_progress=PROGRESS)
            μ_repl = mean(q_avg)
            L_repl = sqrt(cov(q_avg))
            @test μ ≈ μ_repl rtol = 1e-5
            @test L ≈ L_repl rtol = 1e-5
        end
    end
end
