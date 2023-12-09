
const PROGRESS = length(ARGS) > 0 && ARGS[1] == "--progress" ? true : false

using Test

@testset "inference RepGradELBO VILocationScale Bijectors" begin
    @testset "$(modelname) $(objname) $(realtype) $(adbackname)"  for
        realtype ∈ [Float64, Float32],
        (modelname, modelconstr) ∈ Dict(
            :NormalLogNormalMeanField => normallognormal_meanfield,
        ),
        (objname, objective) ∈ Dict(
            :RepGradELBOClosedFormEntropy  => RepGradELBO(10),
            :RepGradELBOStickingTheLanding => RepGradELBO(10, entropy = StickingTheLandingEntropy()),
        ),
        (adbackname, adbackend) ∈ Dict(
            :ForwarDiff  => AutoForwardDiff(),
            :ReverseDiff => AutoReverseDiff(),
            #:Zygote      => AutoZygote(), 
            #:Enzyme      => AutoEnzyme(),
        )

        seed = (0x38bef07cf9cc549d)
        rng  = StableRNG(seed)

        modelstats = modelconstr(rng, realtype)
        @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

        T, η = is_meanfield ? (5_000, 1e-2) : (30_000, 1e-3)

        b    = Bijectors.bijector(model)
        b⁻¹  = inverse(b)
        μ0   = Zeros(realtype, n_dims)
        L0   = Diagonal(Ones(realtype, n_dims))

        q0_η = if is_meanfield
            MeanFieldGaussian(zeros(realtype, n_dims), Diagonal(ones(realtype, n_dims)))
        else
            L0 = Matrix{realtype}(I, n_dims, n_dims) |> LowerTriangular
            FullRankGaussian(zeros(realtype, n_dims), L0)
        end
        q0_z = Bijectors.transformed(q0_η, b⁻¹)

        @testset "convergence" begin
            Δλ₀ = sum(abs2, μ0 - μ_true) + sum(abs2, L0 - L_true)
            q, stats, _ = optimize(
                rng, model, objective, q0_z, T;
                optimizer     = Optimisers.Adam(realtype(η)),
                show_progress = PROGRESS,
                adbackend     = adbackend,
            )

            μ  = q.dist.location
            L  = q.dist.scale
            Δλ = sum(abs2, μ - μ_true) + sum(abs2, L - L_true)

            @test Δλ ≤ Δλ₀/T^(1/4)
            @test eltype(μ) == eltype(μ_true)
            @test eltype(L) == eltype(L_true)
        end

        @testset "determinism" begin
            rng = StableRNG(seed)
            q, stats, _ = optimize(
                rng, model, objective, q0_z, T;
                optimizer     = Optimisers.Adam(realtype(η)),
                show_progress = PROGRESS,
                adbackend     = adbackend,
            )
            μ  = q.dist.location
            L  = q.dist.scale

            rng_repl = StableRNG(seed)
            q, stats, _ = optimize(
                rng_repl, model, objective, q0_z, T;
                optimizer     = Optimisers.Adam(realtype(η)),
                show_progress = PROGRESS,
                adbackend     = adbackend,
            )
            μ_repl = mean(q.dist)
            L_repl = sqrt(cov(q.dist))
            @test μ == μ_repl
            @test L == L_repl
        end
    end
end
