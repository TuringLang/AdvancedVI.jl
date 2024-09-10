
using Test
using Test: @testset, @test

using Base.Iterators
using Bijectors
using Distributions
using FillArrays
using LinearAlgebra
using PDMats
using Pkg
using Random, StableRNGs
using SimpleUnPack: @unpack
using Statistics
using StatsBase

using Functors
using DistributionsAD
@functor TuringDiagMvNormal

using LogDensityProblems
using Optimisers
using ADTypes
using ForwardDiff, ReverseDiff, Zygote, Enzyme

if VERSION >= v"1.10"
    Pkg.add("Tapir")
    using Tapir
end

using AdvancedVI

const GROUP = get(ENV, "GROUP", "All")

# Models for Inference Tests
struct TestModel{M,L,S,SC}
    model::M
    μ_true::L
    L_true::S
    n_dims::Int
    strong_convexity::SC
    is_meanfield::Bool
end
include("models/normal.jl")
include("models/normallognormal.jl")

# Tests
if GROUP == "All" || GROUP == "Interface"
    include("interface/ad.jl")
    include("interface/optimize.jl")
    include("interface/repgradelbo.jl")
    include("interface/rules.jl")
    include("interface/averaging.jl")
    include("interface/location_scale.jl")
end

const PROGRESS = haskey(ENV, "PROGRESS")

if GROUP == "All" || GROUP == "Inference"
    include("inference/repgradelbo_distributionsad.jl")
    include("inference/repgradelbo_locationscale.jl")
    include("inference/repgradelbo_locationscale_bijectors.jl")
end

if GROUP == "GPU"
    if get(ENV, "ADVANCEDVI_TEST_CUDA", "false") == "true"
        Pkg.add("CUDA")
        using CUDA
        include("gpu/cuda.jl")
    end

    if get(ENV, "ADVANCEDVI_TEST_METAL", "false") == "true"
        Pkg.add("Metal")
        using Metal
        if Metal.functional()
            include("gpu/metal.jl")
        end
    end

    if get(ENV, "ADVANCEDVI_TEST_AMDGPU", "false") == "true"
        Pkg.add("AMDGPU")
        using AMDGPU
        if AMDGPU.functional() && AMDGPU.functional(:MIOpen)
            include("gpu/amdgpu.jl")
        end
    end
end
