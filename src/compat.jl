# Copy-paste from Bijectors.jl@0.10
function _forward(d::UnivariateDistribution, x)
    y, logjac = Bijectors.with_logabsdet_jacobian(Identity{0}(), x)
    return (x = x, y = y, logabsdetjac = logjac, logpdf = logpdf.(d, x))
end

forward(rng::Random.AbstractRNG, d::Distribution) = _forward(d, rand(rng, d))
function forward(rng::Random.AbstractRNG, d::Distribution, num_samples::Int)
    return _forward(d, rand(rng, d, num_samples))
end
function _forward(d::Distribution, x)
    y, logjac = Bijectors.with_logabsdet_jacobian(identity, x)
    return (x = x, y = y, logabsdetjac = logjac, logpdf = logpdf(d, x))
end

function _forward(td::Bijectors.TransformedDistribution, x)
    y, logjac = Bijectors.with_logabsdet_jacobian(td.transform, x)
    return (
        x = x,
        y = y,
        logabsdetjac = logjac,
        logpdf = logpdf_forward(td, x, logjac)
    )
end
function forward(rng::Random.AbstractRNG, td::Bijectors.TransformedDistribution)
    return _forward(td, rand(rng, td.dist))
end
function forward(rng::Random.AbstractRNG, td::Bijectors.TransformedDistribution, num_samples::Int)
    return _forward(td, rand(rng, td.dist, num_samples))
end

"""
    forward(d::Distribution)
    forward(d::Distribution, num_samples::Int)

Returns a `NamedTuple` with fields `x`, `y`, `logabsdetjac` and `logpdf`.

In the case where `d isa TransformedDistribution`, this means
- `x = rand(d.dist)`
- `y = d.transform(x)`
- `logabsdetjac` is the logabsdetjac of the "forward" transform.
- `logpdf` is the logpdf of `y`, not `x`

In the case where `d isa Distribution`, this means
- `x = rand(d)`
- `y = x`
- `logabsdetjac = 0.0`
- `logpdf` is logpdf of `x`
"""
forward(d::Distribution) = forward(Random.default_rng(), d)
forward(d::Distribution, num_samples::Int) = forward(Random.default_rng(), d, num_samples)
