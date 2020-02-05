# AdvancedVI.jl
A library for variational Bayesian inference in Julia.

At the time of writing (05/02/2020), implementations of the variational inference (VI) interface and some algorithms are implemented in [Turing.jl](https://github.com/TuringLang/Turing.jl). The idea is to soon separate the VI functionality in Turing.jl out and into this package.

The purpose of this package will then be to provide a common interface together with implementations of standard algorithms and utilities with the goal of ease of use and the ability for other packages, e.g. Turing.jl, to write a light wrapper around AdvancedVI.jl for integration. 

As an example, in Turing.jl we support automatic differentiation variational inference (ADVI) but really the only piece of code tied into the Turing.jl is the conversion of a `Turing.Model` to a `logjoint(z)` function which computes `z ↦ log p(x, z)`, with `x` denoting the observations embedded in the `Turing.Model`. As long as this `logjoint(z)` method is compatible with some AD framework, e.g. `ForwardDiff.jl` or `Zygote.jl`, this is all we need from Turing.jl to be able to perform ADVI!

Stay tuned!

## References

- Jordan, Michael I., Zoubin Ghahramani, Tommi S. Jaakkola, and Lawrence K. Saul. "An introduction to variational methods for graphical models." Machine learning 37, no. 2 (1999): 183-233.
- Blei, David M., Alp Kucukelbir, and Jon D. McAuliffe. "Variational inference: A review for statisticians." Journal of the American statistical Association 112, no. 518 (2017): 859-877.
- Kucukelbir, Alp, Rajesh Ranganath, Andrew Gelman, and David Blei. "Automatic variational inference in Stan." In Advances in Neural Information Processing Systems, pp. 568-576. 2015.
- Salimans, Tim, and David A. Knowles. "Fixed-form variational posterior approximation through stochastic linear regression." Bayesian Analysis 8, no. 4 (2013): 837-882.
- Beal, Matthew James. Variational algorithms for approximate Bayesian inference. 2003.
