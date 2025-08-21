
struct MixedADTestModel end

function LogDensityProblems.logdensity(::MixedADTestModel, θ)
    return sum(abs2, θ)
end

function LogDensityProblems.dimension(::MixedADTestModel)
    return 3
end

function LogDensityProblems.capabilities(::Type{<:MixedADTestModel})
    return LogDensityProblems.LogDensityOrder{1}()
end

# We will check that the deliberately incorrect derivative below
# is returned when differentiating through `LogDensityProblems.logdensity`.
# This would mean that the AD backend is indeed using this (wrong) implementation of
# `logdensity_and_gradient` instead of directly differentiating through `logdensity`.
const EXPECTED_RESULT = [1.0, 2.0, 3.0]

function LogDensityProblems.logdensity_and_gradient(::MixedADTestModel, θ)
    return (Float64(ℯ), EXPECTED_RESULT)
end

function mixedad_test_fwd(x, prob)
    xs = repeat(x, 1, 2)
    return (
        mean(Base.Fix1(LogDensityProblems.logdensity, prob), eachcol(xs)) +
        LogDensityProblems.logdensity(prob, x)
    )/2
end

if AD isa Union{<:AutoReverseDiff,<:AutoZygote,<:AutoEnzyme,<:AutoMooncake}
    @testset "MixedADLogDensityProblem" begin
        model = MixedADTestModel()
        model_ad = AdvancedVI.MixedADLogDensityProblem(model)

        d = 3
        x = ones(Float64, d)

        @testset "interface" begin
            @test LogDensityProblems.dimension(model) ==
                LogDensityProblems.dimension(model_ad)
            @test last(LogDensityProblems.logdensity(model, x)) ≈
                last(LogDensityProblems.logdensity(model_ad, x))
        end

        @testset "custom rrule" begin
            out = DiffResults.DiffResult(0.0, zeros(d))
            AdvancedVI._value_and_gradient!(mixedad_test_fwd, out, AD, x, model_ad)
            @test DiffResults.gradient(out) ≈ EXPECTED_RESULT

            out = DiffResults.DiffResult(0.0, zeros(d))
            prep = AdvancedVI._prepare_gradient(mixedad_test_fwd, AD, x, model_ad)
            AdvancedVI._value_and_gradient!(mixedad_test_fwd, out, prep, AD, x, model_ad)
            @test DiffResults.gradient(out) ≈ EXPECTED_RESULT
        end
    end
end
