
using ReTest
using ForwardDiff, ReverseDiff, Tracker, Enzyme, Zygote
using ADTypes

@testset "ad" begin
    @testset "$(adname)" for (adname, adsymbol) ∈ Dict(
          :ForwardDiff => AutoForwardDiff(),
          :ReverseDiff => AutoReverseDiff(),
          :Zygote      => AutoZygote(),
          :Tracker     => AutoTracker(),
          :Enzyme      => AutoEnzyme(),
        )
        D = 10
        A = randn(D, D)
        λ = randn(D)
        grad_buf = DiffResults.GradientResult(λ)
        AdvancedVI.grad!(adsymbol, λ, grad_buf) do λ′
            λ′'*A*λ′ / 2
        end
        ∇ = DiffResults.gradient(grad_buf)
        f = DiffResults.value(grad_buf)
        @test ∇ ≈ (A + A')*λ/2
        @test f ≈ λ'*A*λ / 2
    end
end
