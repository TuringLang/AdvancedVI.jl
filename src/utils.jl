using Distributions

using Bijectors: Bijectors


function rand_and_logjac(rng::Random.AbstractRNG, dist::Distribution)
    x = rand(rng, dist)
    return x, zero(eltype(x))
end

function rand_and_logjac(rng::Random.AbstractRNG, dist::Bijectors.TransformedDistribution)
    x = rand(rng, dist.dist)
    y, logjac = Bijectors.with_logabsdet_jacobian(dist.transform, x)
    return y, logjac
end

function rand_and_logjac(rng::Random.AbstractRNG, dist::Distribution, n_samples::Int)
    x = rand(rng, dist, n_samples)
    return x, zero(eltype(x))
end

function rand_and_logjac(rng::Random.AbstractRNG, dist::Bijectors.TransformedDistribution, n_samples::Int)
    x = rand(rng, dist.dist, n_samples)
    y, logjac = Bijectors.with_logabsdet_jacobian(dist.transform, x)
    return y, logjac
end

function pm_next!(pm, stats::NamedTuple)
    ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end
