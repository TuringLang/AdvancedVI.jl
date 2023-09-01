
const PROGRESS = length(ARGS) > 0 && ARGS[1] == "--progress" ? true : false

using ReTest

@testset "advi" begin
    @testset "locscale" begin
        @testset "$(modelname) $(objname) $(realtype) $(adbackname)"  for
            realtype ∈ [Float64], # Currently only tested against Float64
            (modelname, modelconstr) ∈ Dict(
                :NormalLogNormalMeanField => normallognormal_meanfield,
            ),
            (objname, objective) ∈ Dict(
                :ADVIClosedFormEntropy  => ADVI(10),
                :ADVIStickingTheLanding => ADVI(10, entropy = StickingTheLandingEntropy()),
            ),
            (adbackname, adbackend) ∈ Dict(
                :ForwarDiff  => AutoForwardDiff(),
                # :ReverseDiff => AutoReverseDiff(),
                # :Zygote      => AutoZygote(), 
                # :Enzyme      => AutoEnzyme(),
            )

            seed = (0x38bef07cf9cc549d)
            rng  = StableRNG(seed)

            modelstats = modelconstr(realtype; rng)
            @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

            T, η = is_meanfield ? (5_000, 1e-2) : (30_000, 1e-3)

            b    = Bijectors.bijector(model)
            b⁻¹  = inverse(b)
            μ₀   = zeros(realtype, n_dims)
            L₀   = Diagonal(ones(realtype, n_dims))

            q₀_η = TuringDiagMvNormal(μ₀, diag(L₀))
            q₀_z = Bijectors.transformed(q₀_η, b⁻¹)

            @testset "convergence" begin
                Δλ₀ = sum(abs2, μ₀ - μ_true) + sum(abs2, L₀ - L_true)
                q, stats, _ = optimize(
                    model, objective, q₀_z, T;
                    optimizer     = Optimisers.Adam(realtype(η)),
                    show_progress = PROGRESS,
                    rng           = rng,
                    adbackend     = adbackend,
                )

                μ  = mean(q.dist)
                L  = sqrt(cov(q.dist))
                Δλ = sum(abs2, μ - μ_true) + sum(abs2, L - L_true)

                @test Δλ ≤ Δλ₀/T^(1/4)
                @test eltype(μ) == eltype(μ_true)
                @test eltype(L) == eltype(L_true)
            end

            @testset "determinism" begin
                rng = StableRNG(seed)
                q, stats, _ = optimize(
                    model, objective, q₀_z, T;
                    optimizer     = Optimisers.Adam(realtype(η)),
                    show_progress = PROGRESS,
                    rng           = rng,
                    adbackend     = adbackend,
                )
                μ  = mean(q.dist)
                L  = sqrt(cov(q.dist))

                rng_repl = StableRNG(seed)
                q, stats, _ = optimize(
                    model, objective, q₀_z, T;
                    optimizer     = Optimisers.Adam(realtype(η)),
                    show_progress = PROGRESS,
                    rng           = rng_repl,
                    adbackend     = adbackend,
                )
                μ_repl = mean(q.dist)
                L_repl = sqrt(cov(q.dist))
                @test μ == μ_repl
                @test L == L_repl
            end
        end
    end
end
