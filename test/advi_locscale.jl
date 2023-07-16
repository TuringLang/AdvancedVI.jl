
const PROGRESS = length(ARGS) > 0 && ARGS[1] == "--progress" ? true : false

using ReTest
using Bijectors
using LogDensityProblems
using Optimisers
using Distributions
using PDMats
using LinearAlgebra
using SimpleUnPack: @unpack

struct TestModel{M,L,S}
    model::M
    μ_true::L
    L_true::S
    n_dims::Int
    is_meanfield::Bool
end

include("models/normallognormal.jl")
include("models/normal.jl")
include("models/utils.jl")

@testset "advi" begin
    @testset "locscale" begin
        @testset "$(modelname) $(objname) $(realtype) $(adbackname)"  for
            realtype ∈ [Float32, Float64],
            (modelname, modelconstr) ∈ Dict(
                :NormalLogNormalMeanField => normallognormal_meanfield,
                :NormalLogNormalFullRank  => normallognormal_fullrank,
                :NormalMeanField          => normal_meanfield,
                :NormalFullRank           => normal_fullrank,
            ),
            (objname, objective) ∈ Dict(
                :ADVIClosedFormEntropy  => (model, b, M) -> ADVI(model, M; b),
                :ADVIStickingTheLanding => (model, b, M) -> ADVI(model, M; b, entropy = StickingTheLandingEntropy()),
                :ADVIFullMonteCarlo     => (model, b, M) -> ADVI(model, M; b, entropy = FullMonteCarloEntropy()),
            ),
            (adbackname, adbackend) ∈ Dict(
                :ForwarDiff  => AutoForwardDiff(),
                # :ReverseDiff => AutoReverseDiff(),
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
                ones(realtype, n_dims) |> Diagonal
            else
                diagm(ones(realtype, n_dims)) |> LowerTriangular
            end
            q₀ = if is_meanfield
                VIMeanFieldGaussian(μ₀, L₀)
            else
                VIFullRankGaussian(μ₀, L₀)
            end

            obj = objective(model, b⁻¹, 10)

            @testset "convergence" begin
                Δλ₀ = sum(abs2, μ₀ - μ_true) + sum(abs2, L₀ - L_true)
                q, stats  = optimize(
                    obj, q₀, T;
                    optimizer = Optimisers.Adam(1e-3),
                    progress  = PROGRESS,
                    rng       = rng,
                    adbackend = adbackend,
                )

                μ  = q.location
                L  = q.scale
                Δλ = sum(abs2, μ - μ_true) + sum(abs2, L - L_true)

                @test Δλ ≤ Δλ₀/√T
                @test eltype(μ) == eltype(μ_true)
                @test eltype(L) == eltype(L_true)
            end

            @testset "determinism" begin
                rng      = Philox4x(UInt64, seed, 8)
                q, stats = optimize(
                    obj, q₀, T;
                    optimizer = Optimisers.Adam(1e-3),
                    progress  = PROGRESS,
                    rng       = rng,
                    adbackend = adbackend,
                )
                μ  = q.location
                L  = q.scale

                rng_repl = Philox4x(UInt64, seed, 8)
                q, stats = optimize(
                    obj, q₀, T;
                    optimizer = Optimisers.Adam(1e-3),
                    progress  = PROGRESS,
                    rng       = rng_repl,
                    adbackend = adbackend,
                )
                μ_repl = q.location
                L_repl = q.scale
                @test μ == μ_repl
                @test L == L_repl
            end
        end
    end
end
