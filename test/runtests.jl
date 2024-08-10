
using Test
using Test: @testset, @test

using Base.Iterators
using Bijectors
using Distributions
using FillArrays
using LinearAlgebra
using PDMats
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
using Enzyme, ForwardDiff, ReverseDiff, Zygote

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
