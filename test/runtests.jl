
using ReTest
using ReTest: @testset, @test

using Comonicon
using Random
using Random123
using Statistics
using Distributions
using LinearAlgebra
using AdvancedVI

include("ad.jl")
include("distributions.jl")
include("advi_locscale.jl")

@main function runtests(patterns...; dry::Bool = false)
    retest(patterns...; dry = dry, verbose = Inf)
end

