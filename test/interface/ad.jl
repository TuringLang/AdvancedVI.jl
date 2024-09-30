
using Test

const interface_ad_backends = Dict(
    :ForwardDiff => AutoForwardDiff(),
    :ReverseDiff => AutoReverseDiff(),
    :Zygote => AutoZygote(),
)

if @isdefined(Tapir)
    interface_ad_backends[:Tapir] = AutoTapir(; safe_mode=false)
end

if @isdefined(Enzyme)
    interface_ad_backends[:Enzyme] = AutoEnzyme()
end

@testset "ad" begin
    @testset "$(adname)" for (adname, adtype) in interface_ad_backends
        D = 10
        A = randn(D, D)
        λ = randn(D)
        b = randn(D)
        grad_buf = DiffResults.GradientResult(λ)
        f(λ′, aux) = λ′' * A * λ′ / 2 + dot(aux.b, λ′)
        AdvancedVI.value_and_gradient!(adtype, f, λ, (b=b,), grad_buf)
        ∇ = DiffResults.gradient(grad_buf)
        f = DiffResults.value(grad_buf)
        @test ∇ ≈ (A + A') * λ / 2 + b
        @test f ≈ λ' * A * λ / 2 + dot(b, λ)
    end
end
