
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
    n_samples::Int,
    q        ::Bijectors.TransformedDistribution,
    q_stop   ::Bijectors.TransformedDistribution,
    ent_est
)
    transform     = q.transform
    q_base        = q.dist
    q_base_stop   = q_stop.dist
    ∑logabsdetjac = 0.0
    base_samples  = rand(rng, q_base, n_samples)
    samples       = mapreduce(hcat, eachcol(base_samples)) do base_sample
        sample, logabsdetjac = with_logabsdet_jacobian(transform, base_sample)
        ∑logabsdetjac       += logabsdetjac
        sample
    end
    entropy_base = AdvancedVI.estimate_entropy_maybe_stl(
        ent_est, base_samples, q_base, q_base_stop
    )
    entropy      = entropy_base + ∑logabsdetjac/n_samples
    samples, entropy
end
end
