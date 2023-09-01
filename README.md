# AdvancedVI.jl
A library for variational Bayesian inference in Julia.

At the time of writing (05/02/2020), implementations of the variational inference (VI) interface and some algorithms are implemented in [Turing.jl](https://github.com/TuringLang/Turing.jl). The idea is to soon separate the VI functionality in Turing.jl out and into this package.

The purpose of this package will then be to provide a common interface together with implementations of standard algorithms and utilities with the goal of ease of use and the ability for other packages, e.g. Turing.jl, to write a light wrapper around AdvancedVI.jl for integration. 

As an example, in Turing.jl we support automatic differentiation variational inference (ADVI) but really the only piece of code tied into the Turing.jl is the conversion of a `Turing.Model` to a `logjoint(z)` function which computes `z ↦ log p(x, z)`, with `x` denoting the observations embedded in the `Turing.Model`. As long as this `logjoint(z)` method is compatible with some AD framework, e.g. `ForwardDiff.jl` or `Zygote.jl`, this is all we need from Turing.jl to be able to perform ADVI!

## [WIP] Interface
- `vi`: the main interface to the functionality in this package
  - `vi(model, alg)`: only used when `alg` has a default variational posterior which it will provide.
  - `vi(model, alg, q::VariationalPosterior, θ)`: `q` represents the family of variational distributions and `θ` is the initial parameters "indexing" the starting distribution. This assumes that there exists an implementation `Variational.update(q, θ)` which returns the variational posterior corresponding to parameters `θ`.
  - `vi(model, alg, getq::Function, θ)`: here `getq(θ)` is a function returning a `VariationalPosterior` corresponding to `θ`.
- `optimize!(vo, alg::VariationalInference{AD}, q::VariationalPosterior, model::Model, θ; optimizer = TruncatedADAGrad())`
- `grad!(vo, alg::VariationalInference, q, model::Model, θ, out, args...)`
  - Different combinations of variational objectives (`vo`), VI methods (`alg`), and variational posteriors (`q`) might use different gradient estimators. `grad!` allows us to specify these different behaviors.

## Examples
### Variational Inference
A very simple generative model is the following

    μ ~ 𝒩(0, 1)
    xᵢ ∼ 𝒩(μ, 1) , ∀i = 1, …, n

where μ and xᵢ are some ℝᵈ vectors and 𝒩 denotes a d-dimensional multivariate Normal distribution.

Given a set of `n` observations `[x₁, …, xₙ]` we're interested in finding the distribution `p(μ∣x₁, …, xₙ)` over the mean `μ`. We can obtain (an approximation to) this distribution that using AdvancedVI.jl!

First we generate some observations and set up the problem:
```julia
julia> using Distributions

julia> d = 2; n = 100;

julia> observations = randn((d, n)); # 100 observations from 2D 𝒩(0, 1)

julia> # Define generative model
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
And finally we can perform VI! The usual inferface is to call `vi` which behind the scenes takes care of the optimization and returns the resulting variational posterior:
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
Unfortunately, the *final* value of the ELBO is not always a very good diagnostic, though the ELBO is an important metric to keep an eye on during training since an *increase* in the ELBO means we're going in the right direction. Luckily, this is such a simple problem that we can indeed obtain a closed form solution! Because we're lazy (at least I am), we'll let [ConjugatePriors.jl](https://github.com/JuliaStats/ConjugatePriors.jl) do this for us:
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

julia> p_samples = rand(true_posterior, 10_000); q_samples = rand(q, 10_000);

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

## Implementing your own training loop
Sometimes it might be convenient to roll your own training loop rather than using `vi(...)`. Here's some psuedo-code for how one would do that when used together with Turing.jl:

```julia
using Turing, AdvancedVI, DiffResults
using Turing: Variational

using ProgressMeter

# Assuming you have an instance of a Turing model (`model`)

# 1. Create log-joint needed for ELBO evaluation
logπ = Variational.make_logjoint(model)

# 2. Define objective
variational_objective = Variational.ELBO()

# 3. Optimizer
optimizer = Variational.DecayedADAGrad()

# 4. VI-algorithm
alg = ADVI(10, 1000)

# 5. Variational distribution
function getq(θ)
    # ...
end

# 6. [OPTIONAL] Implement convergence criterion
function hasconverged(args...)
    # ...
end

# 7. [OPTIONAL] Implement a callback for tracking stats
function callback(args...)
    # ...
end

# 8. Train
converged = false
step = 1

prog = ProgressMeter.Progress(num_steps, 1)

diff_results = DiffResults.GradientResult(θ_init)

while (step ≤ num_steps) && !converged
    # 1. Compute gradient and objective value; results are stored in `diff_results`
    AdvancedVI.grad!(variational_objective, alg, getq, model, diff_results)

    # 2. Extract gradient from `diff_result`
    ∇ = DiffResults.gradient(diff_result)

    # 3. Apply optimizer, e.g. multiplying by step-size
    Δ = apply!(optimizer, θ, ∇)

    # 4. Update parameters
    @. θ = θ - Δ

    # 5. Do whatever analysis you want
    callback(args...)

    # 6. Update
    converged = hasconverged(...) # or something user-defined
    step += 1

    ProgressMeter.next!(prog)
end
```


## References

- Jordan, Michael I., Zoubin Ghahramani, Tommi S. Jaakkola, and Lawrence K. Saul. "An introduction to variational methods for graphical models." Machine learning 37, no. 2 (1999): 183-233.
- Blei, David M., Alp Kucukelbir, and Jon D. McAuliffe. "Variational inference: A review for statisticians." Journal of the American statistical Association 112, no. 518 (2017): 859-877.
- Kucukelbir, Alp, Rajesh Ranganath, Andrew Gelman, and David Blei. "Automatic variational inference in Stan." In Advances in Neural Information Processing Systems, pp. 568-576. 2015.
- Salimans, Tim, and David A. Knowles. "Fixed-form variational posterior approximation through stochastic linear regression." Bayesian Analysis 8, no. 4 (2013): 837-882.
- Beal, Matthew James. Variational algorithms for approximate Bayesian inference. 2003.
