
const PROGRESS = length(ARGS) > 0 && ARGS[1] == "--progress" ? true : false

using Test

@testset "inference RepGradELBO DistributionsAD" begin
    @testset "$(modelname) $(objname) $(realtype) $(adbackname)"  for
        realtype ∈ [Float64, Float32],
        (modelname, modelconstr) ∈ Dict(
            :Normal=> normal_meanfield,
        ),
        (objname, objective) ∈ Dict(
            :RepGradELBOClosedFormEntropy  => RepGradELBO(10),
            :RepGradELBOStickingTheLanding => RepGradELBO(10, entropy = StickingTheLandingEntropy()),
        ),
        (adbackname, adbackend) ∈ Dict(
            :ForwarDiff  => AutoForwardDiff(),
            #:ReverseDiff => AutoReverseDiff(),
            #:Zygote      => AutoZygote(), 
            #:Enzyme      => AutoEnzyme(),
        )

        seed = (0x38bef07cf9cc549d)
        rng  = StableRNG(seed)

        modelstats = modelconstr(rng, realtype)
        @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

        T, η = is_meanfield ? (5_000, 1e-2) : (30_000, 1e-3)

        μ0 = Zeros(realtype, n_dims)
        L0 = Diagonal(Ones(realtype, n_dims))
        q0 = TuringDiagMvNormal(μ0, diag(L0))

        @testset "convergence" begin
            Δλ₀ = sum(abs2, μ0 - μ_true) + sum(abs2, L0 - L_true)
            q, stats, _ = optimize(
                rng, model, objective, q0, T;
                optimizer     = Optimisers.Adam(realtype(η)),
                show_progress = PROGRESS,
                adbackend     = adbackend,
            )

            μ  = mean(q)
            L  = sqrt(cov(q))
            Δλ = sum(abs2, μ - μ_true) + sum(abs2, L - L_true)

            @test Δλ ≤ Δλ₀/T^(1/4)
            @test eltype(μ) == eltype(μ_true)
            @test eltype(L) == eltype(L_true)
        end

        @testset "determinism" begin
            rng = StableRNG(seed)
            q, stats, _ = optimize(
                rng, model, objective, q0, T;
                optimizer     = Optimisers.Adam(realtype(η)),
                show_progress = PROGRESS,
                adbackend     = adbackend,
            )
            μ  = mean(q)
            L  = sqrt(cov(q))

            rng_repl = StableRNG(seed)
            q, stats, _ = optimize(
                rng_repl, model, objective, q0, T;
                optimizer     = Optimisers.Adam(realtype(η)),
                show_progress = PROGRESS,
                adbackend     = adbackend,
            )
            μ_repl = mean(q)
            L_repl = sqrt(cov(q))
            @test μ == μ_repl
            @test L == L_repl
        end
    end
end

