
module AdvancedVIBijectorsExt

if isdefined(Base, :get_extension)
    using AdvancedVI
    using Bijectors
    using Random
else
    using ..AdvancedVI
    using ..Bijectors
    using ..Random
end

function transform_samples_with_jacobian(unconst_samples, transform, n_samples)
    unconst_iter = AdvancedVI.eachsample(unconst_samples)
    unconst_init = first(unconst_iter)

    samples_init, logjac_init = with_logabsdet_jacobian(transform, unconst_init)

    samples_and_logjac = mapreduce(
        AdvancedVI.catsamples_and_acc,
        Iterators.drop(unconst_iter, 1);
        init=(AdvancedVI.samples_expand_dim(samples_init), logjac_init)
    ) do sample
        with_logabsdet_jacobian(transform, sample)
    end
    samples = first(samples_and_logjac)
    logjac  = last(samples_and_logjac)/n_samples
    samples, logjac
end

function AdvancedVI.reparam_with_entropy(
    rng      ::Random.AbstractRNG,
    q        ::Bijectors.TransformedDistribution,
    q_stop   ::Bijectors.TransformedDistribution,
    n_samples::Int,
    ent_est  ::AdvancedVI.AbstractEntropyEstimator
)
    transform       = q.transform
    q_unconst       = q.dist
    q_unconst_stop  = q_stop.dist

    # Draw samples and compute entropy of the uncontrained distribution
    unconst_samples, unconst_entropy = AdvancedVI.reparam_with_entropy(
        rng, q_unconst, q_unconst_stop, n_samples, ent_est
    )

    # Apply bijector to samples while estimating its jacobian
    samples, logjac = transform_samples_with_jacobian(
        unconst_samples, transform, n_samples
    )
    entropy = unconst_entropy + logjac
    samples, entropy
end
end
