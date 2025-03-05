module AdvancedVIBijectorsExt

if isdefined(Base, :get_extension)
    using AdvancedVI
    using Bijectors
    using LinearAlgebra
    using Optimisers
    using Random
else
    using ..AdvancedVI
    using ..Bijectors
    using ..LinearAlgebra
    using ..Optimisers
    using ..Random
end

function AdvancedVI.apply(
    op::ClipScale,
    ::Type{<:Bijectors.TransformedDistribution{<:AdvancedVI.MvLocationScale}},
    params,
    restructure,
)
    return apply_clip_scale(op, params, restructure)
end

function AdvancedVI.apply(
    op::ClipScale,
    ::Type{<:Bijectors.TransformedDistribution{<:AdvancedVI.MvLocationScaleLowRank}},
    params,
    restructure,
)
    return apply_clip_scale(op, params, restructure)
end

function apply_clip_scale(op::ClipScale, params, restructure)
    q = restructure(params)
    ϵ = convert(eltype(params), op.epsilon)

    if isa(q.dist, AdvancedVI.MvLocationScale)
        diag_idx = diagind(q.dist.scale)
        @. q.dist.scale[diag_idx] = max(q.dist.scale[diag_idx], ϵ)
    elseif isa(q.dist, AdvancedVI.MvLocationScaleLowRank)
        @. q.dist.scale_diag = max(q.dist.scale_diag, ϵ)
    else
        error("Unsupported distribution type")
    end

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

    # Draw samples and compute entropy of the uncontrained distribution
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
