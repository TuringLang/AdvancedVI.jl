
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

function AdvancedVI.reparam_with_entropy(
    rng      ::Random.AbstractRNG,
    q        ::Bijectors.TransformedDistribution,
    q_stop   ::Bijectors.TransformedDistribution,
    n_samples::Int,
    ent_est  ::AdvancedVI.AbstractEntropyEstimator
)
    transform    = q.transform
    q_base       = q.dist
    q_base_stop  = q_stop.dist
    base_samples = rand(rng, q_base, n_samples)
    it           = AdvancedVI.eachsample(base_samples)
    sample_init  = first(it)

    samples_and_logjac = mapreduce(
        AdvancedVI.catsamples_and_acc,
        Iterators.drop(it, 1);
        init=with_logabsdet_jacobian(transform, sample_init)
    ) do sample
        with_logabsdet_jacobian(transform, sample)
    end
    samples = first(samples_and_logjac)
    logjac  = last(samples_and_logjac)

    entropy_base = AdvancedVI.estimate_entropy_maybe_stl(
        ent_est, base_samples, q_base, q_base_stop
    )

    entropy = entropy_base + logjac/n_samples
    samples, entropy
end
end
