
const PROGRESS = length(ARGS) > 0 && ARGS[1] == "--progress" ? true : false

using ReTest

@testset "advi" begin
    @testset "locscale" begin
        @testset "$(modelname) $(objname) $(realtype) $(adbackname)"  for
            realtype ∈ [Float64], # Currently only tested against Float64
            (modelname, modelconstr) ∈ Dict(
                :NormalLogNormalMeanField => normallognormal_meanfield,
                :NormalLogNormalFullRank  => normallognormal_fullrank,
                :NormalMeanField          => normal_meanfield,
                :NormalFullRank           => normal_fullrank,
            ),
            (objname, objective) ∈ Dict(
                :ADVIClosedFormEntropy  => (model, b⁻¹, M) -> ADVI(model, M; invbij = b⁻¹),
                :ADVIStickingTheLanding => (model, b⁻¹, M) -> ADVI(model, M; invbij = b⁻¹, entropy = StickingTheLandingEntropy()),
            ),
            (adbackname, adbackend) ∈ Dict(
                :ForwarDiff  => AutoForwardDiff(),
                :ReverseDiff => AutoReverseDiff(),
                # :Zygote      => AutoZygote(), 
                # :Enzyme      => AutoEnzyme(),
            )

            seed = (0x38bef07cf9cc549d, 0x49e2430080b3f797)
            rng  = Philox4x(UInt64, seed, 8)

            T = 10000
            modelstats = modelconstr(realtype; rng)
            @unpack model, μ_true, L_true, n_dims, is_meanfield = modelstats

            b    = Bijectors.bijector(model)
            b⁻¹  = inverse(b)

            μ₀ = zeros(realtype, n_dims)
            L₀ = if is_meanfield
                FillArrays.Eye(n_dims) |> Diagonal
            else
                FillArrays.Eye(n_dims) |> Matrix |> LowerTriangular
            end

            q₀ = if is_meanfield
                VIMeanFieldGaussian(μ₀, L₀)
            else
                VIFullRankGaussian(μ₀, L₀)
            end

            obj = objective(model, b⁻¹, 10)

            @testset "convergence" begin
                Δλ₀         = sum(abs2, μ₀ - μ_true) + sum(abs2, L₀ - L_true)
                q, stats, _ = optimize(
                    obj, q₀, T;
                    optimizer     = Optimisers.Adam(1e-2),
                    show_progress = PROGRESS,
                    rng           = rng,
                    adbackend     = adbackend,
                )

                μ  = q.location
                L  = q.scale
                Δλ = sum(abs2, μ - μ_true) + sum(abs2, L - L_true)

                @test Δλ ≤ Δλ₀/T^(1/4)
                @test eltype(μ) == eltype(μ_true)
                @test eltype(L) == eltype(L_true)
            end

            @testset "determinism" begin
                rng         = Philox4x(UInt64, seed, 8)
                q, stats, _ = optimize(
                    obj, q₀, T;
                    optimizer     = Optimisers.Adam(realtype(1e-2)),
                    show_progress = PROGRESS,
                    rng           = rng,
                    adbackend     = adbackend,
                )
                μ  = q.location
                L  = q.scale

                rng_repl    = Philox4x(UInt64, seed, 8)
                q, stats, _ = optimize(
                    obj, q₀, T;
                    optimizer     = Optimisers.Adam(realtype(1e-2)),
                    show_progress = PROGRESS,
                    rng           = rng_repl,
                    adbackend     = adbackend,
                )
                μ_repl = q.location
                L_repl = q.scale
                @test μ == μ_repl
                @test L == L_repl
            end
        end
    end
end
