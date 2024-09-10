
# [Subsampling](@id subsampling)

## Introduction
For problems with large datasets, evaluating the objective may become computationally too expensive.
In this regime, many variational inference algorithms can readily incorporate datapoint subsampling to reduce the per-iteration computation cost[^HBWP2013][^TL2014].
Notice that many variational objectives require only *gradients* of the log target.
In a lot of cases, the gradient can be replaced with an *unbiased estimate* of the log target.
This section describes how to do this in `AdvancedVI`.


[^HBWP2013]: Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). Stochastic variational inference. *Journal of Machine Learning Research*.
[^TL2014]: Titsias, M., & Lázaro-Gredilla, M. (2014, June). Doubly stochastic variational Bayes for non-conjugate inference. In *International Conference on Machine Learning.*

## API
Subsampling is performed by wrapping the desired variational objective with the following objective:

```@docs
Subsampled
```
Furthermore, the target distribution `prob` must implement the following function:
```@docs
AdvancedVI.subsample
```
The subsampling strategy used by `Subsampled` is what is known as "random reshuffling".
That is, the full dataset is shuffled and then partitioned into batches.
The batches are picked one at a time in a "sampling without replacement" fashion, which results in faster convergence than independently subsampling batches.[^KKMG2024]

[^KKMG2024]: Kim, K., Ko, J., Ma, Y., & Gardner, J. R. (2024). Demystifying SGD with Doubly Stochastic Gradients. In *International Conference on Machine Learning.*

!!! note
    For the log target to be an valid unbiased estimate of the full batch gradient, the average over the batch must be adjusted by a constant factor ``n/b``, where ``n`` is the number of datapoints and ``b``  is the size of the minibatch (`length(batch)`). See the [example](@ref subsampling_example) for a demonstration of how to do this.
    

## [Example](@id subsampling)

We will consider a sum of multivariate Gaussians, and subsample over the components of the sum:

```@example subsampling
using SimpleUnPack, LogDensityProblems, Distributions, Random, LinearAlgebra

struct SubsampledMvNormals{D <: MvNormal, F <: Real}
    dists::Vector{D}
    likeadj::F
end

function SubsampledMvNormals(rng::Random.AbstractRNG, n_dims, n_normals::Int)
    μs = randn(rng, n_dims, n_normals)
    Σ = I
    dists = MvNormal.(eachcol(μs), Ref(Σ))
    SubsampledMvNormals{eltype(dists), Float64}(dists, 1.0)
end

function LogDensityProblems.logdensity(m::SubsampledMvNormals, x)
    @unpack likeadj, dists = m
    likeadj*mapreduce(Base.Fix2(logpdf, x), +, dists)
end
```

Notice that, when computing the log-density, we multiple by a constant `likeadj`.
This is to adjust the strength of the likelihood when minibatching is used.

To use subsampling, we need to implement `subsample`, where we also compute the likelihood adjustment `likeadj`:
```@example subsampling
using AdvancedVI

function AdvancedVI.subsample(m::SubsampledMvNormals, idx)
    n_data = length(m.dists)
    SubsampledMvNormals(m.dists[idx], n_data/length(idx))
end
```

The objective is constructed as follows:
```@example subsampling
n_dims = 10
n_data = 1024
prob = SubsampledMvNormals(Random.default_rng(), n_dims, n_data);
```
We will a dataset with `1024` datapoints.

For the objective, we will use `RepGradELBO`.
To apply subsampling, it suffices to wrap with `subsampled`:
```@example subsampling
batchsize = 8
full_obj = RepGradELBO(1)
sub_obj = Subsampled(full_obj, batchsize, 1:n_data);
```
We can now invoke `optimize` to perform inference.
```@setup subsampling
using ForwardDiff, ADTypes, Optimisers, Plots

Σ_true = Diagonal(fill(1/n_data, n_dims))
μ_true = mean([mean(component) for component in prob.dists])
Σsqrt_true = sqrt(Σ_true)

q0 = MeanFieldGaussian(zeros(n_dims), Diagonal(ones(n_dims)))

adtype = AutoForwardDiff()
optimizer = Adam(0.01)
averager = PolynomialAveraging()

function callback(; averaged_params, restructure, kwargs...)
    q = restructure(averaged_params)
    μ, Σ = mean(q), cov(q)
    dist2 = sum(abs2, μ - μ_true) + tr(Σ + Σ_true - 2*sqrt(Σsqrt_true*Σ*Σsqrt_true))
    (dist = sqrt(dist2),)
end

n_iters = 3*10^2
_, q, stats_full, _ = optimize(
    prob, full_obj, q0, n_iters; optimizer, averager, show_progress=false, adtype, callback,
)

n_iters = 10^3
_, _, stats_sub, _ = optimize(
    prob, sub_obj, q0, n_iters; optimizer, averager, show_progress=false, adtype, callback,
)

x = [stat.iteration for stat in stats_full]
y = [stat.dist for stat in stats_full]
Plots.plot(x, y, xlabel="Iterations", ylabel="Wasserstein-2 Distance", label="Full Batch")

x = [stat.iteration for stat in stats_sub]
y = [stat.dist for stat in stats_sub]
Plots.plot!(x, y, xlabel="Iterations", ylabel="Wasserstein-2 Distance", label="Subsampling (Random Reshuffling)")
savefig("subsampling_iteration.svg")

x = [stat.elapsed_time for stat in stats_full]
y = [stat.dist for stat in stats_full]
Plots.plot(x, y, xlabel="Wallclock Time (sec)", ylabel="Wasserstein-2 Distance", label="Full Batch")

x = [stat.elapsed_time for stat in stats_sub]
y = [stat.dist for stat in stats_sub]
Plots.plot!(x, y, xlabel="Wallclock Time (sec)", ylabel="Wasserstein-2 Distance", label="Subsampling (Random Reshuffling)")
savefig("subsampling_wallclocktime.svg")
```
Let's first compare the convergence of full-batch `RepGradELBO` versus subsampled `RepGradELBO` with respect to the number of iterations:

![](subsampling_iteration.svg)

While it seems that subsampling results in slower convergence, the real power of subsampling is revealed when comparing with respect to the wallclock time:

![](subsampling_wallclocktime.svg)

Clearly, subsampling results in a vastly faster convergence speed.
