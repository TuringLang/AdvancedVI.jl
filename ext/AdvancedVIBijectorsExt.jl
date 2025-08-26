module AdvancedVIBijectorsExt

using AdvancedVI
using DiffResults: DiffResults
using Bijectors
using LinearAlgebra
using Optimisers
using Random

function AdvancedVI.init(
    rng::Random.AbstractRNG,
    alg::AdvancedVI.ParamSpaceSGD,
    q_init::Bijectors.TransformedDistribution,
    prob,
)
    (; adtype, optimizer, averager, objective, operator) = alg
    if q_init.dist isa AdvancedVI.MvLocationScale &&
        operator isa AdvancedVI.IdentityOperator
        @warn(
            "IdentityOperator is used with a variational family <:MvLocationScale. Optimization can easily fail under this combination due to singular scale matrices. Consider using the operator `ClipScale` instead.",
            typeof(q_init),
            typeof(q_init.dist),
            typeof(operator)
        )
    end
    params, re = Optimisers.destructure(q_init)
    opt_st = Optimisers.setup(optimizer, params)
    obj_st = AdvancedVI.init(rng, objective, adtype, q_init, prob, params, re)
    avg_st = AdvancedVI.init(averager, params)
    grad_buf = DiffResults.DiffResult(zero(eltype(params)), similar(params))
    return AdvancedVI.ParamSpaceSGDState(prob, q_init, 0, grad_buf, opt_st, obj_st, avg_st)
end

function AdvancedVI.apply(
    op::ClipScale,
    ::Type{<:Bijectors.TransformedDistribution{<:AdvancedVI.MvLocationScale}},
    state,
    params,
    restructure,
)
    q = restructure(params)
    ϵ = convert(eltype(params), op.epsilon)

    # Project the scale matrix to the set of positive definite triangular matrices
    diag_idx = diagind(q.dist.scale)
    @. q.dist.scale[diag_idx] = max(q.dist.scale[diag_idx], ϵ)

    params, _ = Optimisers.destructure(q)

    return params
end

function AdvancedVI.apply(
    op::ClipScale,
    ::Type{<:Bijectors.TransformedDistribution{<:AdvancedVI.MvLocationScaleLowRank}},
    state,
    params,
    restructure,
)
    q = restructure(params)
    ϵ = convert(eltype(params), op.epsilon)

    @. q.dist.scale_diag = max(q.dist.scale_diag, ϵ)

    params, _ = Optimisers.destructure(q)

    return params
end

function AdvancedVI.apply(
    ::AdvancedVI.ProximalLocationScaleEntropy,
    ::Type{<:Bijectors.TransformedDistribution{<:AdvancedVI.MvLocationScale}},
    leaf::Optimisers.Leaf{<:Union{<:DoG,<:DoWG,<:Descent},S},
    params,
    restructure,
) where {S}
    q = restructure(params)

    stepsize = AdvancedVI.stepsize_from_optimizer_state(leaf.rule, leaf.state)
    diag_idx = diagind(q.dist.scale)
    scale_diag = q.dist.scale[diag_idx]
    @. q.dist.scale[diag_idx] =
        scale_diag + 1 / 2 * (sqrt(scale_diag^2 + 4 * stepsize) - scale_diag)

    params, _ = Optimisers.destructure(q)

    return params
end

function AdvancedVI.reparam_with_entropy(
    rng::Random.AbstractRNG,
    q::Bijectors.TransformedDistribution,
    q_stop::Bijectors.TransformedDistribution,
    n_samples::Int,
    ent_est::AdvancedVI.AbstractEntropyEstimator,
)
    transform = q.transform
    q_unconst = q.dist
    q_unconst_stop = q_stop.dist

    # Draw samples and compute entropy of the unconstrained distribution
    unconstr_samples, unconst_entropy = AdvancedVI.reparam_with_entropy(
        rng, q_unconst, q_unconst_stop, n_samples, ent_est
    )

    # Apply bijector to samples while estimating its jacobian
    unconstr_iter = AdvancedVI.eachsample(unconstr_samples)
    unconstr_init = first(unconstr_iter)
    samples_init, logjac_init = with_logabsdet_jacobian(transform, unconstr_init)
    samples_and_logjac = mapreduce(
        AdvancedVI.catsamples_and_acc,
        Iterators.drop(unconstr_iter, 1);
        init=(reshape(samples_init, (:, 1)), logjac_init),
    ) do sample
        with_logabsdet_jacobian(transform, sample)
    end
    samples = first(samples_and_logjac)
    logjac = last(samples_and_logjac) / n_samples

    entropy = unconst_entropy + logjac
    return samples, entropy
end

end
