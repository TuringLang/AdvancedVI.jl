
using ReTest: @testset, @test
#using Random
#using Statistics
#using Distributions, DistributionsAD

println("Environment variables for testing")
println(ENV)

include("ad.jl")
include("distributions.jl")

