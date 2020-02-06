# AdvancedVI.jl
A library for variational Bayesian inference in Julia.

At the time of writing (05/02/2020), implementations of the variational inference (VI) interface and some algorithms are implemented in [Turing.jl](https://github.com/TuringLang/Turing.jl). The idea is to soon separate the VI functionality in Turing.jl out and into this package.

The purpose of this package will then be to provide a common interface together with implementations of standard algorithms and utilities with the goal of ease of use and the ability for other packages, e.g. Turing.jl, to write a light wrapper around AdvancedVI.jl for integration. 

As an example, in Turing.jl we support automatic differentiation variational inference (ADVI) but really the only piece of code tied into the Turing.jl is the conversion of a `Turing.Model` to a `logjoint(z)` function which computes `z ↦ log p(x, z)`, with `x` denoting the observations embedded in the `Turing.Model`. As long as this `logjoint(z)` method is compatible with some AD framework, e.g. `ForwardDiff.jl` or `Zygote.jl`, this is all we need from Turing.jl to be able to perform ADVI!

## Examples
### Variational Inference
A very simple generative model is the following

    μ ~ 𝒩(0, 1)
    xᵢ ∼ 𝒩(μ, 1) , ∀i = 1, …, n

where μ and xᵢ are some ℝᵈ vectors and 𝒩 denotes a d-dimensional multivariate Normal distribution.

Given a set of `n` observations `[x₁, …, xₙ]` we're interested in finding the distribution `p(μ∣x₁, …, xₙ)` over the mean `μ`. Let's do that using AdvancedVI.jl!

First we generate some observations and set up the problem:
```julia
julia> using Distributions

julia> d = 2; n = 100;

julia> observations = randn((d, n)); # 100 observations from 2D 𝒩(0, 1)

julia> # Define generative model
       #
       #    μ ~ 𝒩(0, 1)
       #    xᵢ ∼ 𝒩(μ, 1) , ∀i = 1, …, n
       prior(μ) = logpdf(MvNormal(ones(d)), μ)
prior (generic function with 1 method)

julia> likelihood(x, μ) = sum(logpdf(MvNormal(μ, ones(d)), x))
likelihood (generic function with 1 method)

julia> logπ(μ) = likelihood(observations, μ) + prior(μ)
logπ (generic function with 1 method)

julia> logπ(randn(2))  # <= just checking that it works
-311.74132761437653
```
Now there are mainly two different ways of specifying the approximate posterior (and its family). The first is by providing a mapping from distribution parameters to the distribution `θ ↦ q(⋅∣θ)`:
```julia
julia> using DistributionsAD, AdvancedVI

julia> # Using a function z ↦ q(⋅∣z)
       getq(θ) = TuringDiagMvNormal(θ[1:d], exp.(θ[d + 1:4]))
getq (generic function with 1 method)
```
Then we make the choice of algorithm, a subtype of `VariationalInference`, 
```julia
julia> # Perform VI
       advi = ADVI(10, 10_000)
ADVI{AdvancedVI.ForwardDiffAD{40}}(10, 10000)
```
And finally we perform VI!
```julia
julia> q = vi(logπ, advi, getq, randn(4))
[ADVI] Optimizing...100% Time: 0:00:01
TuringDiagMvNormal{Array{Float64,1},Array{Float64,1}}(m=[0.16282745378074515, 0.15789310089462574], σ=[0.09519377533754399, 0.09273176907111745])
```
Let's have a look at the resulting ELBO:
```julia
julia> AdvancedVI.elbo(advi, q, logπ, 1000)
-287.7866366886285
```
Unfortunately, the *final* value of the ELBO is not always a very good diagnostic, though it's an important metric to keep an eye on during training since a *decrease* in the ELBO means we're going in the right direction. Luckily, this is such a simple problem that we can indeed obtain a closed form solution! Because we're lazy (at least I am), we'll let [ConjugatePriors.jl](https://github.com/JuliaStats/ConjugatePriors.jl) do this for us:
```julia
julia> # True posterior
       using ConjugatePriors

julia> pri = MvNormal(zeros(2), ones(2));

julia> true_posterior = posterior((pri, pri.Σ), MvNormal, observations)
DiagNormal(
dim: 2
μ: [0.1746546592601148, 0.16457110079543008]
Σ: [0.009900990099009901 0.0; 0.0 0.009900990099009901]
)
```
Comparing to our variational approximation, this looks pretty good! Worth noting that in this particular case the variational posterior seems to overestimate the variance.

To conclude, let's make a somewhat pretty picture:
```julia
julia> using Plots

julia> p_samples = rand(true_posterior, 10_000)
2×10000 Array{Float64,2}:
 0.226544  0.0386895  0.363355  0.103075  0.17044   0.200926  0.211849  0.118225   0.151113  0.282503  0.211417  0.449449  0.0954799  …   0.220084   0.210435  0.090151  0.0574122  0.188425  0.293998  0.209506  0.154152  0.054414  0.300582  0.267328  0.173011  0.257673
 0.129798  0.0209997  0.268326  0.171766  0.198101  0.274679  0.102984  0.0241527  0.147919  0.143214  0.136427  0.137116  0.196594      -0.0891732  0.238403  0.275577  0.279917   0.181441  0.148274  0.136249  0.263142  0.156751  0.263506  0.192837  0.206643  0.201236

julia> q_samples = rand(q, 10_000)
2×10000 Array{Float64,2}:
 0.128835  0.0671971  0.235282  0.203082  0.325733  0.210107  0.277108   0.124131   0.144778   0.260337  0.130576  0.145427  0.233502  …  -0.00986131  0.0461744  0.109007  0.285839  0.168117  0.279098  0.290743  0.158525  0.188196  0.169235  0.0837522  0.252854 
 0.193174  0.101739   0.18992   0.297838  0.244713  0.253424  0.0807829  0.0133219  0.0180343  0.340928  0.14168   0.190034  0.255748      0.22417     0.117033   0.235965  0.318793  0.143454  0.147184  0.181239  0.283337  0.251981  0.212938  0.33779    0.0493925

julia> p1 = histogram(p_samples[1, :], label="p"); histogram!(q_samples[1, :], alpha=0.7, label="q")

julia> title!(raw"$\mu_1$")

julia> p2 = histogram(p_samples[2, :], label="p"); histogram!(q_samples[2, :], alpha=0.7, label="q")

julia> title!(raw"$\mu_2$")

julia> plot(p1, p2)
```
![Histogram](hist.png?raw=true)

### Simple example: using Advanced.jl to directly minimize the KL-divergence between two distributions `p(z)` and `q(z)`
In VI we aim to approximate the true posterior `p(z ∣ x)` by some approximate variational posterior `q(z)` by maximizing the ELBO:

    ELBO(q) = 𝔼_q[log p(x, z) - log q(z)]

Observe that we can express the ELBO as the negative KL-divergence between `p(x, ⋅)` and `q(⋅)`:

    ELBO(q) = - 𝔼_q[log (q(z) / p(x, z))]
            = - KL(q(⋅) || p(x, ⋅))

So if we apply VI to something that isn't an actual posterior, i.e. there's no data involved and we write `p(z ∣ x) = p(z)`, we're really just minimizing the KL-divergence between the distributions.

Therefore, we can try out `AdvancedVI.jl` real quick by applying using the interface to minimize the KL-divergence between two distributions:

```julia
julia> using Distributions, DistributionsAD, AdvancedVI

julia> # Target distribution
       p = MvNormal(ones(2))
ZeroMeanDiagNormal(
dim: 2
μ: [0.0, 0.0]
Σ: [1.0 0.0; 0.0 1.0]
)

julia> logπ(z) = logpdf(p, z)
logπ (generic function with 1 method)

julia> # Make a choice of VI algorithm
       advi = ADVI(10, 1000)
ADVI{AdvancedVI.ForwardDiffAD{40}}(10, 1000)
```
Now there are two different ways of specifying the approximate posterior (and its family); the first is by providing a mapping from parameters to distribution `θ ↦ q(⋅∣θ)`:
```julia
julia> # Using a function z ↦ q(⋅∣z)
       getq(θ) = TuringDiagMvNormal(θ[1:2], exp.(θ[3:4]))
getq (generic function with 1 method)

julia> # Perform VI
       q = vi(logπ, advi, getq, randn(4))
┌ Info: [ADVI] Should only be seen once: optimizer created for θ
└   objectid(θ) = 0x5ddb564423896704
[ADVI] Optimizing...100% Time: 0:00:01
TuringDiagMvNormal{Array{Float64,1},Array{Float64,1}}(m=[-0.012691337868985757, -0.0004442434543332919], σ=[1.0334797673569802, 0.9957355128767893])
```
Or we can check the ELBO (which in this case since, as mentioned, doesn't involve data, is the negative KL-divergence):
```julia
julia> AdvancedVI.elbo(advi, q, logπ, 1000)  # empirical estimate
0.08031049170093245
```
It's worth noting that the actual value of the ELBO doesn't really tell us too much about the quality of fit. In this particular case, because we're *directly* minimizing the KL-divergence, we can only say something useful if we reach 0, in which case we have obtained the true distribution.

Let's just quickly check the mean-squared error between the `log p(z)` and `log q(z)` for a random set of samples from the target `p`:
```julia
julia> zs = rand(p, 100);

julia> mean(abs2, logpdf(q, zs) - logpdf(p, zs))
0.0014889109427524852
```
That doesn't look too bad!


## References

- Jordan, Michael I., Zoubin Ghahramani, Tommi S. Jaakkola, and Lawrence K. Saul. "An introduction to variational methods for graphical models." Machine learning 37, no. 2 (1999): 183-233.
- Blei, David M., Alp Kucukelbir, and Jon D. McAuliffe. "Variational inference: A review for statisticians." Journal of the American statistical Association 112, no. 518 (2017): 859-877.
- Kucukelbir, Alp, Rajesh Ranganath, Andrew Gelman, and David Blei. "Automatic variational inference in Stan." In Advances in Neural Information Processing Systems, pp. 568-576. 2015.
- Salimans, Tim, and David A. Knowles. "Fixed-form variational posterior approximation through stochastic linear regression." Bayesian Analysis 8, no. 4 (2013): 837-882.
- Beal, Matthew James. Variational algorithms for approximate Bayesian inference. 2003.
