
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
                :ADVIClosedFormEntropy  => (model, b⁻¹, M) -> ADVI(model, M; invbij = b⁻¹),
                :ADVIStickingTheLanding => (model, b⁻¹, M) -> ADVI(model, M; invbij = b⁻¹, entropy = StickingTheLandingEntropy()),
            ),
            (adbackname, adbackend) ∈ Dict(
                :ForwarDiff  => AutoForwardDiff(),
                # :ReverseDiff => AutoReverseDiff(),
                # :Zygote      => AutoZygote(), 
                # :Enzyme      => AutoEnzyme(),
            )

            seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
            rng  = Philox4x(UInt64, seed, 8)

            modelstats = modelconstr(realtype; rng)
            @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

            T, η = is_meanfield ? (5_000, 1e-2) : (30_000, 1e-3)

            b    = Bijectors.bijector(model)
            b⁻¹  = inverse(b)
            μ₀   = zeros(realtype, n_dims)
	    L₀   = Diagonal(ones(realtype, n_dims))

	    q₀ = TuringDiagMvNormal(μ₀, diag(L₀))

            obj = objective(model, b⁻¹, 10)

            @testset "convergence" begin
                Δλ₀ = sum(abs2, μ₀ - μ_true) + sum(abs2, L₀ - L_true)
                q, stats, _, _ = optimize(
                    obj, q₀, T;
                    optimizer     = Optimisers.Adam(realtype(η)),
                    show_progress = PROGRESS,
                    rng           = rng,
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
                rng = Philox4x(UInt64, seed, 8)
                q, stats, _, _ = optimize(
                    obj, q₀, T;
                    optimizer     = Optimisers.Adam(realtype(η)),
                    show_progress = PROGRESS,
                    rng           = rng,
                    adbackend     = adbackend,
                )
		μ  = mean(q)
		L  = sqrt(cov(q))

                rng_repl = Philox4x(UInt64, seed, 8)
                q, stats, _, _ = optimize(
                    obj, q₀, T;
                    optimizer     = Optimisers.Adam(realtype(η)),
                    show_progress = PROGRESS,
                    rng           = rng_repl,
                    adbackend     = adbackend,
                )
		μ_repl = mean(q)
		L_repl = sqrt(cov(q))
                @test μ == μ_repl
                @test L == L_repl
            end
        end
    end
end
