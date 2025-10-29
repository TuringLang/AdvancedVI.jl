
struct SubsampledNormals{D<:Normal,F<:Real,C}
    dists::Vector{D}
    likeadj::F
    capability::C
end

function SubsampledNormals(rng::Random.AbstractRNG, n_normals::Int, capability)
    μs = randn(rng, n_normals)
    σs = ones(n_normals)
    dists = Normal.(μs, σs)
    return SubsampledNormals{eltype(dists),Float64,typeof(capability)}(
        dists, 1.0, capability
    )
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

function LogDensityProblems.dimension(::SubsampledNormals)
    return 1
end

function LogDensityProblems.capabilities(::Type{SubsampledNormals{D,F,C}}) where {D,F,C}
    return C()
end

function AdvancedVI.subsample(m::SubsampledNormals, idx)
    n_data = length(m.dists)
    return SubsampledNormals(m.dists[idx], n_data/length(idx), m.capability)
end

function subsamplednormal(rng::Random.AbstractRNG, n_data::Int; capability::Int=1)
    cap = if capability == 1
        LogDensityProblems.LogDensityOrder{1}()
    elseif capability == 2
        LogDensityProblems.LogDensityOrder{2}()
    else
        LogDensityProblems.LogDensityOrder{0}()
    end
    model = SubsampledNormals(rng, n_data, cap)
    n_dims = 1
    μ_true = [mean([mean(dist) for dist in model.dists])]
    L_true = Diagonal([sqrt(1/n_data)])
    return TestModel(model, μ_true, L_true, n_dims, 1, true)
end
