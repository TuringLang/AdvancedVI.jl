
using Test
using Test: @testset, @test

using ADTypes
using Base.Iterators
using Bijectors
using DiffResults
using Distributions
using FillArrays
using ForwardDiff, ReverseDiff
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD
using Optimisers
using PDMats
using Pkg
using Random, StableRNGs
using Statistics
using StatsBase

using AdvancedVI

AD_str = get(ENV, "AD", "ReverseDiff")

const AD = if AD_str == "ReverseDiff"
    AutoReverseDiff()
elseif AD_str == "Mooncake"
    using Mooncake
    AutoMooncake(; config=Mooncake.Config())
elseif AD_str == "Zygote"
    using Zygote
    AutoZygote()
elseif AD_str == "Enzyme"
    using Enzyme
    AutoEnzyme()
end

const PROGRESS = haskey(ENV, "PROGRESS")
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

if GROUP == "All" || GROUP == "General"
    include("general/optimize.jl")
    include("general/rules.jl")
    include("general/averaging.jl")
    include("general/clip_scale.jl")
    include("general/proximal_location_scale_entropy.jl")
end

if GROUP == "All" || GROUP == "Families"
    include("families/location_scale.jl")
    include("families/location_scale_low_rank.jl")
end

if GROUP == "All" || GROUP == "General" || GROUP == "AD"
    include("general/ad.jl")
end

if GROUP == "All" || GROUP == "ParamSpaceSGD" || GROUP == "AD"
    include("algorithms/paramspacesgd/repgradelbo.jl")
    include("algorithms/paramspacesgd/scoregradelbo.jl")
    include("algorithms/paramspacesgd/repgradelbo_locationscale.jl")
    include("algorithms/paramspacesgd/repgradelbo_locationscale_bijectors.jl")
    include("algorithms/paramspacesgd/repgradelbo_proximal_locationscale.jl")
    include("algorithms/paramspacesgd/repgradelbo_proximal_locationscale_bijectors.jl")
    include("algorithms/paramspacesgd/scoregradelbo_locationscale.jl")
    include("algorithms/paramspacesgd/scoregradelbo_locationscale_bijectors.jl")
end
