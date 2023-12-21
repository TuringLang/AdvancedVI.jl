```@meta
CurrentModule = AdvancedVI
```

# AdvancedVI

## Introduction
[AdvancedVI](https://github.com/TuringLang/AdvancedVI.jl) provides implementations of variational Bayesian inference (VI) algorithms.
VI algorithms perform scalable and computationally efficient Bayesian inference at the cost of asymptotic exactness.
`AdvancedVI` is part of the [Turing](https://turinglang.org/stable/) probabilistic programming ecosystem.

## Provided Algorithms
`AdvancedVI` currently provides the following algorithm for evidence lower bound maximization:
- [Black-Box Variational Inference](@ref bbvi)
