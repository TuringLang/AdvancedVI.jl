
AD_scoregradelbo_distributionsad = Dict(
    :ForwarDiff => AutoForwardDiff(),
    #:ReverseDiff => AutoReverseDiff(), # DistributionsAD doesn't support ReverseDiff at the moment
    :Zygote => AutoZygote(),
)

if @isdefined(Mooncake)
    AD_scoregradelbo_distributionsad[:Mooncake] = AutoMooncake(; config=Mooncake.Config())
end

#if @isdefined(Enzyme)
#    AD_scoregradelbo_distributionsad[:Enzyme] = AutoEnzyme()
#end

@testset "inference ScoreGradELBO DistributionsAD" begin
    @testset "$(modelname) $(objname) $(realtype) $(adbackname)" for realtype in
                                                                     [Float64, Float32],
        (modelname, modelconstr) in Dict(:Normal => normal_meanfield),
        n_montecarlo in [1, 10],
        (objname, objective) in Dict(
            :ScoreGradELBOClosedFormEntropy => ScoreGradELBO(n_montecarlo),
            :ScoreGradELBOStickingTheLanding =>
                ScoreGradELBO(n_montecarlo; entropy=StickingTheLandingEntropy()),
        ),
        (adbackname, adtype) in AD_scoregradelbo_distributionsad

        seed = (0x38bef07cf9cc549d)
        rng = StableRNG(seed)

        modelstats = modelconstr(rng, realtype)
        @unpack model, μ_true, L_true, n_dims, strong_convexity, is_meanfield = modelstats

        T = 1000
        η = 1e-5
        opt = Optimisers.Descent(realtype(η))

        # For small enough η, the error of SGD, Δλ, is bounded as
        #     Δλ ≤ ρ^T Δλ0 + O(η),
        # where ρ = 1 - ημ, μ is the strong convexity constant.
        contraction_rate = 1 - η * strong_convexity

        μ0 = zeros(realtype, n_dims)
        L0 = Diagonal(ones(realtype, n_dims))
        q0 = TuringDiagMvNormal(μ0, diag(L0))

        @testset "convergence" begin
            Δλ0 = sum(abs2, μ0 - μ_true) + sum(abs2, L0 - L_true)
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

            μ = mean(q_avg)
            L = sqrt(cov(q_avg))
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
            μ = mean(q_avg)
            L = sqrt(cov(q_avg))

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
            μ_repl = mean(q_avg)
            L_repl = sqrt(cov(q_avg))
            @test μ ≈ μ_repl rtol = 1e-5
            @test L ≈ L_repl rtol = 1e-5
        end
    end
end