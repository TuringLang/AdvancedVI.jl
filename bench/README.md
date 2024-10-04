# AdvancedVI.jl Continuous Benchmarking

This subdirectory contains code for continuous benchmarking of the performance of `AdvancedVI.jl`.
The initial version was heavily inspired by the setup of [Lux.jl](https://github.com/LuxDL/Lux.jl/tree/main).
The Github action and pages integration is provided by  https://github.com/benchmark-action/github-action-benchmark/ and [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl).

To run the benchmarks locally, follow the following steps:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.develop("AdvancedVI")
include("benchmarks.jl")
```
