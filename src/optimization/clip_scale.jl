
"""
    ClipScale(ϵ = 1e-5)

Projection operator ensuring that an `MvLocationScale` or `MvLocationScaleLowRank` has a scale with eigenvalues larger than `ϵ`.
`ClipScale` also supports by operating on `MvLocationScale` and `MvLocationScaleLowRank` wrapped by a `Bijectors.TransformedDistribution` object. 
"""
Optimisers.@def struct ClipScale <: AbstractOperator
    epsilon = 1e-5
end

function operate(op::ClipScale, family::Type, params, restructure)
    return error("`ClipScale` is not defined for the variational family of type $(family).")
end

function operate(op::ClipScale, ::Type{<:MvLocationScale}, params, restructure)
    q = restructure(params)
    ϵ = convert(eltype(params), op.epsilon)

    # Project the scale matrix to the set of positive definite triangular matrices
    diag_idx = diagind(q.scale)
    @. q.scale[diag_idx] = max(q.scale[diag_idx], ϵ)

    params, _ = Optimisers.destructure(q)

    return params
end

function operate(op::ClipScale, ::Type{<:MvLocationScaleLowRank}, params, restructure)
    q = restructure(params)
    ϵ = convert(eltype(params), op.epsilon)

    # Project the scale matrix to the set of positive definite triangular matrices
    @. q.scale_diag = max(q.scale_diag, ϵ)

    params, _ = Optimisers.destructure(q)

    return params
end
