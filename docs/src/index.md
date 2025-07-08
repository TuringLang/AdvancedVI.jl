```@meta
CurrentModule = AdvancedVI
```

# AdvancedVI

[AdvancedVI](https://github.com/TuringLang/AdvancedVI.jl) provides implementations of variational Bayesian inference (VI) algorithms.
VI algorithms perform scalable and computationally efficient Bayesian inference at the cost of asymptotic exactness.
`AdvancedVI` is part of the [Turing](https://turinglang.org/stable/) probabilistic programming ecosystem.

# List of Algorithms

  - [ParamSpaceSGD](@ref paramspacesgd)
  - [KLMinRepGradDescent](@ref klminrepgraddescent) (alias of `ADVI`)
  - [KLMinRepGradProxDescent](@ref klminrepgradproxdescent)
  - [KLMinScoreGradDescent](@ref klminscoregraddescent)  (alias of `BBVI`)
