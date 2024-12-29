
function simulate_sequence_average(realtype::Type{<:Real}, avg::AdvancedVI.AbstractAverager)
    d = 3
    n = 10
    xs = randn(realtype, d, n)
    xs_it = eachcol(xs)
    st = AdvancedVI.init(avg, first(xs_it))
    for x in xs_it
        st = AdvancedVI.apply(avg, st, x)
    end
    return AdvancedVI.value(avg, st), xs
end

@testset "averaging" begin
    avg = NoAveraging()
    @testset "$(avg) $(realtype)" for realtype in [Float32, Float64]
        x_avg, xs = simulate_sequence_average(realtype, avg)

        @test eltype(x_avg) == realtype
        @test x_avg ≈ xs[:, end]
    end

    η = 1
    avg = PolynomialAveraging(η)
    @testset "$(avg) $(realtype)" for realtype in [Float32, Float64]
        x_avg, xs = simulate_sequence_average(realtype, avg)

        T = size(xs, 2)
        α = map(1:T) do t
            # Formula from the proof of Theorem 4 by Shamir & Zhang (2013)
            (η + 1) / (t + η) * (t == T ? 1 : prod(j -> (j - 1) / (j + η), (t + 1):T))
        end
        x_avg_true = xs * α

        @test eltype(x_avg) == realtype
        @test x_avg ≈ x_avg_true
    end
end
