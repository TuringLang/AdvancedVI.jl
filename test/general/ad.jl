
@testset "ad" begin
    @testset "value_and_gradient!" begin
        D = 10
        A = randn(D, D)
        λ = randn(D)
        b = randn(D)
        grad_buf = DiffResults.GradientResult(λ)
        f(λ′, aux) = λ′' * A * λ′ / 2 + dot(aux.b, λ′)
        AdvancedVI._value_and_gradient!(f, grad_buf, AD, λ, (b=b,))
        ∇ = DiffResults.gradient(grad_buf)
        f = DiffResults.value(grad_buf)
        @test ∇ ≈ (A + A') * λ / 2 + b
        @test f ≈ λ' * A * λ / 2 + dot(b, λ)
    end

    @testset "value_and_gradient! with prep" begin
        D = 10
        λ = randn(D)
        A = randn(D, D)
        grad_buf = DiffResults.GradientResult(λ)

        b_prep = randn(D)
        f(λ′, aux) = λ′' * A * λ′ / 2 + dot(aux.b, λ′)
        prep = AdvancedVI._prepare_gradient(f, AD, λ, (b=b_prep,))

        b = randn(D)
        AdvancedVI._value_and_gradient!(f, grad_buf, prep, AD, λ, (b=b,))

        ∇ = DiffResults.gradient(grad_buf)
        f = DiffResults.value(grad_buf)
        @test ∇ ≈ (A + A') * λ / 2 + b
        @test f ≈ λ' * A * λ / 2 + dot(b, λ)
    end
end
