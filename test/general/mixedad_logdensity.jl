
AD_mixedad = if TEST_GROUP == "Enzyme"
    Dict(
        :Enzyme => AutoEnzyme(;
            mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation=Enzyme.Const,
        ),
    )
else
    Dict(
        :ReverseDiff => AutoReverseDiff(),
        :Zygote => AutoZygote(),
        :Mooncake => AutoMooncake(; config=Mooncake.Config()),
    )
end

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

function LogDensityProblems.logdensity_and_gradient(::MixedADTestModel, θ)
    return (Float64(ℯ), [1.0, 2.0, 3.0])
end

function mixedad_test_fwd(x, prob)
    xs = repeat(x, 1, 2)
    return (
        mean(Base.Fix1(LogDensityProblems.logdensity, prob), eachcol(xs)) +
        LogDensityProblems.logdensity(prob, x)
    )/2
end

@testset "MixedADLogDensityProblem" begin
    model = MixedADTestModel()
    model_ad = AdvancedVI.MixedADLogDensityProblem(model)

    d = 3
    x = ones(Float64, d)

    @testset "interface" begin
        @test LogDensityProblems.dimension(model) == LogDensityProblems.dimension(model_ad)
        @test last(LogDensityProblems.logdensity(model, x)) ≈
            last(LogDensityProblems.logdensity(model_ad, x))
    end

    @testset "rrule under $(adname)" for (adname, adtype) in AD_mixedad
        out = DiffResults.DiffResult(0.0, zeros(d))
        AdvancedVI._value_and_gradient!(mixedad_test_fwd, out, adtype, x, model_ad)
        @test DiffResults.gradient(out) ≈ [1.0, 2.0, 3.0]

        out = DiffResults.DiffResult(0.0, zeros(d))
        prep = AdvancedVI._prepare_gradient(mixedad_test_fwd, adtype, x, model_ad)
        AdvancedVI._value_and_gradient!(mixedad_test_fwd, out, prep, adtype, x, model_ad)
        @test DiffResults.gradient(out) ≈ [1.0, 2.0, 3.0]
    end
end
