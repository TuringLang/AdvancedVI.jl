
AD_locationscale = if VERSION >= v"1.10"
    Dict(
        :ForwarDiff => AutoForwardDiff(),
        :ReverseDiff => AutoReverseDiff(),
        :Zygote => AutoZygote(),
        :Enzyme => AutoEnzyme(),
    )
else
    Dict(
        :ForwarDiff => AutoForwardDiff(),
        :ReverseDiff => AutoReverseDiff(),
        :Zygote => AutoZygote(),
    )
end

@testset "inference RepGradELBO VILocationScale" begin
    @testset "$(modelname) $(objname) $(realtype) $(adbackname)" for realtype in
                                                                     [Float64, Float32],
        (modelname, modelconstr) in
        Dict(:Normal => normal_meanfield, :Normal => normal_fullrank),
        n_montecarlo in [1, 10],
        (objname, objective) in Dict(
            :RepGradELBOClosedFormEntropy => RepGradELBO(n_montecarlo),
            :RepGradELBOStickingTheLanding =>
                RepGradELBO(n_montecarlo; entropy=StickingTheLandingEntropy()),
        ),
        (adbackname, adtype) in AD_locationscale

        seed = (0x38bef07cf9cc549d)
        rng = StableRNG(seed)

        modelstats = modelconstr(rng, realtype)
        @unpack model, μ_true, L_true, n_dims, strong_convexity, is_meanfield = modelstats

        T = 1000
        η = 1e-3
        opt = Optimisers.Descent(realtype(η))

        # For small enough η, the error of SGD, Δλ, is bounded as
        #     Δλ ≤ ρ^T Δλ0 + O(η),
        # where ρ = 1 - ημ, μ is the strong convexity constant.
        contraction_rate = 1 - η * strong_convexity

        q0 = if is_meanfield
            MeanFieldGaussian(zeros(realtype, n_dims), Diagonal(ones(realtype, n_dims)))
        else
            L0 = LowerTriangular(Matrix{realtype}(I, n_dims, n_dims))
            FullRankGaussian(zeros(realtype, n_dims), L0)
        end

        @testset "convergence" begin
            Δλ0 = sum(abs2, q0.location - μ_true) + sum(abs2, q0.scale - L_true)
            q_avg, _, stats, _ = optimize(
                rng,
                model,
                objective,
                q0,
                T;
                optimizer=opt,
                show_progress=PROGRESS,
                adtype=adtype,
            )

            μ = q_avg.location
            L = q_avg.scale
            Δλ = sum(abs2, μ - μ_true) + sum(abs2, L - L_true)

            @test Δλ ≤ contraction_rate^(T / 2) * Δλ0
            @test eltype(μ) == eltype(μ_true)
            @test eltype(L) == eltype(L_true)
        end

        @testset "determinism" begin
            rng = StableRNG(seed)
            q_avg, _, stats, _ = optimize(
                rng,
                model,
                objective,
                q0,
                T;
                optimizer=opt,
                show_progress=PROGRESS,
                adtype=adtype,
            )
            μ = q_avg.location
            L = q_avg.scale

            rng_repl = StableRNG(seed)
            q_avg, _, stats, _ = optimize(
                rng_repl,
                model,
                objective,
                q0,
                T;
                optimizer=opt,
                show_progress=PROGRESS,
                adtype=adtype,
            )
            μ_repl = q_avg.location
            L_repl = q_avg.scale
            @test μ == μ_repl
            @test L == L_repl
        end
    end
end
