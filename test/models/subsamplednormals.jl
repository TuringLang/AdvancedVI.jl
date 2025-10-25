
struct SubsampledNormals{D<:Normal,F<:Real,C}
    dists::Vector{D}
    likeadj::F
    cap::C
end

function SubsampledNormals(rng::Random.AbstractRNG, n_normals::Int)
    μs = randn(rng, n_normals)
    σs = ones(n_normals)
    dists = Normal.(μs, σs)
    return SubsampledNormals{eltype(dists),Float64}(dists, 1.0)
end

function LogDensityProblems.logdensity(m::SubsampledNormals, x)
    (; likeadj, dists) = m
    return likeadj*mapreduce(Base.Fix2(logpdf, only(x)), +, dists)
end

function LogDensityProblems.logdensity_and_gradient(m::SubsampledNormals, x)
    return (
        LogDensityProblems.logdensity(m, x),
        ForwardDiff.gradient(Base.Fix1(LogDensityProblems.logdensity, m), x),
    )
end

function LogDensityProblems.capabilities(::Type{SubsampledNormals{D,F,C}}) where {D,F,C}
    return C()
end

function AdvancedVI.subsample(m::SubsampledNormals, idx)
    n_data = length(m.dists)
    return SubsampledNormals(m.dists[idx], n_data/length(idx))
end

function subsamplednormal(n_data::Int; capability::Int=1)
    cap = if capability == 1
        LogDensityProblems.LogDensityOrder{1}()
    elseif capability == 2
        LogDensityProblems.LogDensityOrder{2}()
    else
        LogDensityProblems.LogDensityOrder{0}()
    end
    model = SubsampledNormals(Random.default_rng(), n_data, cap)

    n_dims = 1
    μ_true = [mean([mean(dist) for dist in prob.dists])]
    L_true = Diagonal(ones(1))
    return TestModel(model, μ_true, L_true, n_dims, 1, true)
end
