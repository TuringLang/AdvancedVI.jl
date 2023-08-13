
using ReTest
using ForwardDiff, ReverseDiff, Enzyme, Zygote
using ADTypes

@testset "ad" begin
    @testset "$(adname)" for (adname, adsymbol) ∈ Dict(
          :ForwardDiff => AutoForwardDiff(),
          :ReverseDiff => AutoReverseDiff(),
          :Zygote      => AutoZygote(),
          # :Enzyme      => AutoEnzyme(), # Currently not tested against.
        )
        D = 10
        A = randn(D, D)
        λ = randn(D)
        grad_buf = DiffResults.GradientResult(λ)
        f(λ′) = λ′'*A*λ′ / 2
        AdvancedVI.value_and_gradient!(adsymbol, f, λ, grad_buf)
        ∇ = DiffResults.gradient(grad_buf)
        f = DiffResults.value(grad_buf)
        @test ∇ ≈ (A + A')*λ/2
        @test f ≈ λ'*A*λ / 2
    end
end
