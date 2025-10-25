
struct SubsampledNormals{D<:Normal,F<:Real}
    dists::Vector{D}
    likeadj::F
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

function LogDensityProblems.logdensity_gradient_and_hessian(m::SubsampledNormals, x)
    return (
        LogDensityProblems.logdensity(m, x),
        ForwardDiff.gradient(Base.Fix1(LogDensityProblems.logdensity, m), x),
        ForwardDiff.hessian(Base.Fix1(LogDensityProblems.logdensity, m), x),
    )
end

function LogDensityProblems.capabilities(::Type{<:SubsampledNormals})
    return LogDensityProblems.LogDensityOrder{2}()
end

function AdvancedVI.subsample(m::SubsampledNormals, idx)
    n_data = length(m.dists)
    return SubsampledNormals(m.dists[idx], n_data/length(idx))
end

function subsamplednormal(n_data::Int)
    model = SubsampledNormals(Random.default_rng(), n_data)
    n_dims = 1
    μ_true = [mean([mean(dist) for dist in prob.dists])]
    L_true = Diagonal(ones(1))
    return TestModel(model, μ_true, L_true, n_dims, 1, true)
end
