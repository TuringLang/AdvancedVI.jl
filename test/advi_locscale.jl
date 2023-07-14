
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

@testset "exact" begin
    @testset "$(modelname) $(objname) $(realtype)"  for
        realtype ∈ [Float32, Float64],
        (modelname, modelconstr) ∈ Dict(
            :NormalLogNormalMeanField => normallognormal_meanfield,
            :NormalLogNormalFullRank  => normallognormal_fullrank,
        ),
        (objname, objective) ∈ Dict(
            :ADVIClosedFormEntropy  => (model, b⁻¹, M) -> ADVI(model, b⁻¹,                              M),
            :ADVIStickingTheLanding => (model, b⁻¹, M) -> ADVI(model, b⁻¹, StickingTheLandingEntropy(), M),
            :ADVIFullMonteCarlo     => (model, b⁻¹, M) -> ADVI(model, b⁻¹, MonteCarloEntropy(),         M),
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
                optimizer = Optimisers.AdaGrad(1e-1),
                progress  = PROGRESS,
                rng       = rng,
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
                optimizer = Optimisers.AdaGrad(1e-1),
                progress  = PROGRESS,
                rng       = rng,
            )
            μ  = q.location
            L  = q.scale

            rng_repl = Philox4x(UInt64, seed, 8)
            q, stats = optimize(
                obj, q₀, T;
                optimizer = Optimisers.AdaGrad(1e-1),
                progress  = PROGRESS,
                rng       = rng_repl,
            )
            μ_repl = q.location
            L_repl = q.scale
            @test μ == μ_repl
            @test L == L_repl
        end
    end
end

