using AdvancedVI
using Bijectors
using Distributions
using Flux
using ForwardDiff
using LinearAlgebra
using Random
using Test

# include("optimisers.jl")

@testset "AdvancedVI" begin
    @testset "algorithms" begin
        include(joinpath("algorithms", "advi.jl"))
        include(joinpath("algorithms", "bbvi.jl"))
    end
    @testset "distributions" begin
        include(joinpath("distributions", "distributions.jl"))
        include(joinpath("distributions", "diagmvnormal.jl"))
        include(joinpath("distributions", "cholmvnormal.jl"))
    end
    include("gradients.jl")
    include("interface.jl")
    # include("optimisers.jl") # Relying on Tracker...
    include("objectives.jl")
    include("utils.jl")
end
