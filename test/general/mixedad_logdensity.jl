
struct MixedADTestModel end

function LogDensityProblems.logdensity(::MixedADTestModel, θ)
    return Float64(ℯ)
end

function LogDensityProblems.dimension(::MixedADTestModel)
    return 3
end

function LogDensityProblems.capabilities(::Type{<:MixedADTestModel})
    return LogDensityProblems.LogDensityOrder{2}()
end

function LogDensityProblems.logdensity_and_gradient(::MixedADTestModel, θ)
    return (Float64(ℯ), [1.0, 2.0, 3.0])
end

function LogDensityProblems.logdensity_gradient_and_hessian(::MixedADTestModel, θ)
    return (Float64(ℯ), [1.0, 2.0, 3.0], [1.0 1.0 1.0; 2.0 2.0 2.0; 3.0 3.0 3.0])
end

@testset "interface MixedADLogDensityProblem" begin
    model = MixedADTestModel()
    model_ad = AdvancedVI.MixedADLogDensityProblem(model)

    d = 3
    x = ones(Float64, d)

    @test LogDensityProblems.dimension(model) == LogDensityProblems.dimension(model_ad)
    @test LogDensityProblems.capabilities(typeof(model)) ≈
        LogDensityProblems.capabilities(typeof(model_ad))
    @test last(LogDensityProblems.logdensity(model, x)) ≈
        last(LogDensityProblems.logdensity(model_ad, x))
    @test last(LogDensityProblems.logdensity_and_gradient(model, x)) ≈
        last(LogDensityProblems.logdensity_and_gradient(model_ad, x))
    @test last(LogDensityProblems.logdensity_gradient_and_hessian(model, x)) ≈
        last(LogDensityProblems.logdensity_gradient_and_hessian(model_ad, x))
end
