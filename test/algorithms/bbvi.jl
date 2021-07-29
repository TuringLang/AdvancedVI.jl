@testset "bbvi" begin
    using AdvancedVI: CholMvNormal, DiagMvNormal
    ## Testing no transform
    target = MvNormal(ones(2))
    xs = rand(target, 10)
    logπ(z) = logpdf(target, z)
    qs = [
        CholMvNormal(randn(2), LowerTriangular(randn(2, 2))),
        DiagMvNormal(randn(2), randn(2)),
    ]
    advi = BBVI(10, 1000)
    for q in qs
        q = vi(logπ, advi, q; opt=ADAM(0.01))
        @test mean(abs2, logpdf(q, xs) - logpdf(target, xs)) ≤ 0.05
    end

    ## Testing with transform # TODO

end
