
using Test
using Test: @testset, @test

using ADTypes
using Base.Iterators
using Bijectors
using DiffResults
using Distributions
using FillArrays
using ForwardDiff, Zygote
using LinearAlgebra
using LogDensityProblems
using Optimisers
using PDMats
using Pkg
using Random, StableRNGs
using Statistics
using StatsBase

using AdvancedVI

AD_str = get(ENV, "AD", "ReverseDiff")

const AD = if AD_str == "ForwardDiff"
    AutoForwardDiff()
elseif AD_str == "ReverseDiff"
    using ReverseDiff
    AutoReverseDiff()
elseif AD_str == "Mooncake"
    using Mooncake
    AutoMooncake(; config=Mooncake.Config())
elseif AD_str == "Zygote"
    AutoZygote()
elseif AD_str == "Enzyme"
    using Enzyme
    AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Const,
    )
end

const PROGRESS = haskey(ENV, "PROGRESS")
const TEST_GROUP = get(ENV, "TEST_GROUP", "All")

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

if TEST_GROUP == "All" || TEST_GROUP == "General"
    # Interface tests that do not involve testing on Enzyme
    include("general/optimize.jl")
    include("general/rules.jl")
    include("general/averaging.jl")
    include("general/clip_scale.jl")
    include("general/proximal_location_scale_entropy.jl")
    include("general/mixedad_logdensity.jl")
end

if TEST_GROUP == "All" || TEST_GROUP == "General" || TEST_GROUP == "AD"
    # Interface tests that involve testing on Enzyme
    include("general/ad.jl")
end

if TEST_GROUP == "All" || TEST_GROUP == "Families"
    include("families/location_scale.jl")
    include("families/location_scale_low_rank.jl")
end

if TEST_GROUP == "All" || TEST_GROUP == "ParamSpaceSGD" || TEST_GROUP == "AD"
    if AD isa Union{<:AutoReverseDiff, <:AutoZygote, <:AutoMooncake}
        include("algorithms/paramspacesgd/repgradelbo.jl")
        include("algorithms/paramspacesgd/repgradelbo_locationscale.jl")
        include("algorithms/paramspacesgd/repgradelbo_locationscale_bijectors.jl")
        include("algorithms/paramspacesgd/repgradelbo_proximal_locationscale.jl")
        include("algorithms/paramspacesgd/repgradelbo_proximal_locationscale_bijectors.jl")
    end
    if AD isa Union{<:AutoReverseDiff, <:AutoZygote, <:AutoMooncake, <:AutoForwardDiff}
        include("algorithms/paramspacesgd/scoregradelbo.jl")
        include("algorithms/paramspacesgd/scoregradelbo_locationscale.jl")
        include("algorithms/paramspacesgd/scoregradelbo_locationscale_bijectors.jl")
    end
end
