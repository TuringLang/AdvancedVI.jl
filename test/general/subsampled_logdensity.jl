struct SecondOrderLogDensity end

function LogDensityProblems.logdensity(::SecondOrderLogDensity, x)
    return -sum(abs2, x) / 2
end

function LogDensityProblems.logdensity_and_gradient(::SecondOrderLogDensity, x)
    return -sum(abs2, x) / 2, -x
end

function LogDensityProblems.logdensity_gradient_and_hessian(::SecondOrderLogDensity, x)
    return -sum(abs2, x) / 2, -x, -Matrix{eltype(x)}(I, length(x), length(x))
end

LogDensityProblems.dimension(::SecondOrderLogDensity) = 2

function LogDensityProblems.capabilities(::Type{SecondOrderLogDensity})
    return LogDensityProblems.LogDensityOrder{2}()
end

struct VariableDimensionLogDensity
    dimension::Int
end

LogDensityProblems.dimension(prob::VariableDimensionLogDensity) = prob.dimension

function LogDensityProblems.capabilities(::Type{VariableDimensionLogDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

@testset "SubsampledLogDensity" begin
    scales = Float64[]
    make_prob = (_, scale) -> begin
        push!(scales, scale)
        return SecondOrderLogDensity()
    end
    prob = SubsampledLogDensity(SecondOrderLogDensity(), make_prob, 10)

    @test LogDensityProblems.capabilities(typeof(prob)) ==
        LogDensityProblems.LogDensityOrder{2}()

    prob_sub = AdvancedVI.subsample(prob, 1:5)
    @test scales == [2.0]
    @test LogDensityProblems.logdensity_gradient_and_hessian(prob_sub, zeros(2)) ==
        (0.0, zeros(2), -Matrix{Float64}(I, 2, 2))

    @test_throws ArgumentError SubsampledLogDensity(SecondOrderLogDensity(), make_prob, 0)
    @test_throws ArgumentError AdvancedVI.subsample(prob, Int[])
    @test scales == [2.0]

    variable_prob = SubsampledLogDensity(
        VariableDimensionLogDensity(2),
        (batch, _) -> VariableDimensionLogDensity(length(batch)),
        10,
    )
    @test_throws DimensionMismatch AdvancedVI.subsample(variable_prob, 1:3)
end
