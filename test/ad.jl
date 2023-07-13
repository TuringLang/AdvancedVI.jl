
using ReTest
using ForwardDiff, ReverseDiff, Tracker, Enzyme, Zygote
using AdvancedVI: grad!

@testset "ad" begin
    @testset "$(string(adsymbol))" for adsymbol ∈ [
        :forwarddiff, :reversediff, :tracker, :enzyme, :zygote]
        D = 10
        A = randn(D, D)
        λ = randn(D)
        AdvancedVI.setadbackend(adsymbol)
        grad_buf = DiffResults.GradientResult(λ)
        AdvancedVI.grad!(AdvancedVI.ADBackend(), λ, grad_buf) do λ′
            λ′'*A*λ′ / 2
        end
        ∇ = DiffResults.gradient(grad_buf)
        f = DiffResults.value(grad_buf)
        @test ∇ ≈ (A + A')*λ/2
        @test f ≈ λ'*A*λ / 2
    end
end
