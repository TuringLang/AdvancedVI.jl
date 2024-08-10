
@testset "rules" begin
    @testset "$(rule) $(realtype)" for rule in [DoWG(), DoG(), COCOB()],
        realtype in [Float32, Float64]

        T = 10^4

        d = 10
        n = 1000
        w = randn(realtype, d)
        X = rand(realtype, n, d)
        w_true = randn(realtype, d)
        loss(x, w) = mean((x * w .- x * w_true) .^ 2)
        l0 = loss(X, w)

        opt_st = Optimisers.setup(rule, w)
        for t in 1:T
            i = sample(1:n)
            xi = X[i:i, :]
            g = ForwardDiff.gradient(Base.Fix1(loss, xi), w)
            opt_st, w = Optimisers.update!(opt_st, w, g)
        end

        @test eltype(w) == realtype
        @test loss(X, w) < l0 / 10
    end
end
