
using Test

AD_interface = if TEST_GROUP == "Enzyme"
    Dict(
        :Enzyme => AutoEnzyme(;
            mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation=Enzyme.Const,
        ),
    )
else
    Dict(
        :ForwarDiff => AutoForwardDiff(),
        :ReverseDiff => AutoReverseDiff(),
        :Zygote => AutoZygote(),
        :Mooncake => AutoMooncake(; config=Mooncake.Config()),
    )
end

@testset "ad" begin
    @testset "$(adname)" for (adname, adtype) in AD_interface
        D = 10
        A = randn(D, D)
        λ = randn(D)
        b = randn(D)
        grad_buf = DiffResults.GradientResult(λ)
        f(λ′, aux) = λ′' * A * λ′ / 2 + dot(aux.b, λ′)
        AdvancedVI._value_and_gradient!(f, grad_buf, adtype, λ, (b=b,))
        ∇ = DiffResults.gradient(grad_buf)
        f = DiffResults.value(grad_buf)
        @test ∇ ≈ (A + A') * λ / 2 + b
        @test f ≈ λ' * A * λ / 2 + dot(b, λ)
    end

    @testset "$(adname) with prep" for (adname, adtype) in AD_interface
        D = 10
        λ = randn(D)
        A = randn(D, D)
        grad_buf = DiffResults.GradientResult(λ)

        b_prep = randn(D)
        f(λ′, aux) = λ′' * A * λ′ / 2 + dot(aux.b, λ′)
        prep = AdvancedVI._prepare_gradient(f, adtype, λ, (b=b_prep,))

        b = randn(D)
        AdvancedVI._value_and_gradient!(f, grad_buf, prep, adtype, λ, (b=b,))

        ∇ = DiffResults.gradient(grad_buf)
        f = DiffResults.value(grad_buf)
        @test ∇ ≈ (A + A') * λ / 2 + b
        @test f ≈ λ' * A * λ / 2 + dot(b, λ)
    end
end
