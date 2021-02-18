@testset "utils" begin
    using AdvancedVI: XXt, XtX, makelogπ, evaluate
    A = randn(3, 3)
    @test XXt(A) == A * A'
    @test XtX(A) == A' * A

    f(x) = 2x
    make_f(h) = f
    @test makelogπ(f, nothing) == f
    @test makelogπ(make_f, []) == f

    x = rand(2)
    q = MvNormal(ones(2))
    q̂ = transformed(q, Bijectors.RadialLayer(2))
    logπ(x) = logpdf(q, x)
    z, logj = forward(q̂.transform, x)
    @test evaluate(logπ, q, x) == logπ(x)
    @test evaluate(logπ, q̂, x) == logπ(z) + logj
end