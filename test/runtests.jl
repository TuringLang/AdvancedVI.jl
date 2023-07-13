
using Comonicon
using ReTest: @testset, @test
using Random
using Random: default_rng
using Statistics
using Distributions
using LinearAlgebra
using AdvancedVI

const GROUP = get(ENV, "AHMC_TEST_GROUP", "AdvancedHMC")

include("ad.jl")
include("distributions.jl")
include("exact.jl")

@main function runtests(patterns...; dry::Bool = false)
    retest(patterns...; dry = dry, verbose = Inf)
end

