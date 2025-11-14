```@meta
CurrentModule = AdvancedVI
```

# AdvancedVI

[AdvancedVI](https://github.com/TuringLang/AdvancedVI.jl) provides implementations of variational Bayesian inference (VI) algorithms.
VI algorithms perform scalable and computationally efficient Bayesian inference at the cost of asymptotic exactness.
`AdvancedVI` is part of the [Turing](https://turinglang.org/stable/) probabilistic programming ecosystem.

For general usage, first refer to the following page:

  - [General Usage](@ref general)

For using the algorithms implemented in `AdvancedVI`, refer to the corresponding documentations below:

  - [KLMinRepGradDescent](@ref klminrepgraddescent) (alias of `ADVI`)
  - [KLMinRepGradProxDescent](@ref klminrepgradproxdescent)
  - [KLMinScoreGradDescent](@ref klminscoregraddescent)  (alias of `BBVI`)
  - [KLMinNaturalGradDescent](@ref klminnaturalgraddescent)
  - [KLMinSqrtNaturalGradDescent](@ref klminsqrtnaturalgraddescent)
  - [KLMinWassFwdBwd](@ref klminwassfwdbwd)
