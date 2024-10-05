# [Optimization](@id optim)

## Parameter-Free Optimization Rules

We provide custom optimization rules that are not provided out-of-the-box by [Optimisers.jl](https://github.com/FluxML/Optimisers.jl).
The main theme of the provided optimizers is that they are parameter-free.
This means that these optimization rules shouldn't require (or barely) any tuning to obtain performance competitive with well-tuned alternatives.

```@docs
DoG
DoWG
COCOB
```

## Parameter Averaging Strategies

In some cases, the best optimization performance is obtained by averaging the sequence of parameters generated by the optimization algorithm.
For instance, the `DoG`[^IHC2023] and `DoWG`[^KMJ2024] papers report their best performance through averaging.
The benefits of parameter averaging have been specifically confirmed for ELBO maximization[^DCAMHV2020].

```@docs
NoAveraging
PolynomialAveraging
```

[^DCAMHV2020]: Dhaka, A. K., Catalina, A., Andersen, M. R., Magnusson, M., Huggins, J., & Vehtari, A. (2020). Robust, accurate stochastic optimization for variational inference. Advances in Neural Information Processing Systems, 33, 10961-10973.