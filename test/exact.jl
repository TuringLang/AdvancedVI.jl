
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

include("exact/normallognormal.jl")

@testset "exact" begin
    @testset "$(modelname) $(realtype)"  for
        realtype ∈ [Float32, Float64],
        (modelname, modelconstr) ∈ Dict(
            :NormalLogNormalMeanField => normallognormal_meanfield,
            :NormalLogNormalFullRank  => normallognormal_fullrank,
        )

        T = 10000
        modelstats = modelconstr(realtype)
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
            AdvancedVI.VIMeanFieldGaussian(μ₀, L₀, realtype(1e-8))
        else
            AdvancedVI.VIFullRankGaussian(μ₀, L₀, realtype(1e-8))
        end

        Δλ₀ = sum(abs2, μ₀ - μ_true) + sum(abs2, L₀ - L_true)

        objective = AdvancedVI.ADVI(model, b⁻¹, 10)
        q, stats  = AdvancedVI.optimize(
            objective, q₀, T;
            optimizer = Optimisers.AdaGrad(1e-1),
            progress  = PROGRESS,
        )

        μ  = q.location
        L  = q.scale
        Δλ = sum(abs2, μ - μ_true) + sum(abs2, L - L_true)

        @test Δλ ≤ Δλ₀/√T
    end
end
