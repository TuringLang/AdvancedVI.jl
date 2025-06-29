# [Score Gradient Estimator](@id scoregradelbo)

## Overview

The `ScoreGradELBO` implements the score gradient estimator[^G1990][^KR1996][^RSU1996][^W1992] of the ELBO gradient, also known as the score function method and the REINFORCE gradient.
For variational inference, the use of the score gradient was proposed in [^WW2013][^RGB2014].
Unlike the [reparameterization gradient](@ref repgradelbo), the score gradient does not require the target log density to be differentiable, and does not differentiate through the sampling process of the variational approximation $q$.
Instead, it only requires gradients of the log density $\log q$.
For this reason, the score gradient is the standard method to deal with discrete variables and targets with discrete support.

[^G1990]: Glynn, P. W. (1990). Likelihood ratio gradient estimation for stochastic systems. Communications of the ACM, 33(10), 75-84.
[^KR1996]: Kleijnen, J. P., & Rubinstein, R. Y. (1996). Optimization and sensitivity analysis of computer simulation models by the score function method. European Journal of Operational Research, 88(3), 413-427.
[^RSU1996]: Rubinstein, R. Y., Shapiro, A., & Uryasev, S. (1996). The score function method. Encyclopedia of Management Sciences, 1363-1366.
[^W1992]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8, 229-256.
[^WW2013]: Wingate, D., & Weber, T. (2013). Automated variational inference in probabilistic programming. arXiv preprint arXiv:1301.1299.
[^RGB2014]: Ranganath, R., Gerrish, S., & Blei, D. (2014). Black box variational inference. In Artificial intelligence and statistics (pp. 814-822). PMLR.
    In more detail, the score gradient uses the Fisher log-derivative identity: For any regular $f$,
```math
\nabla_{\lambda} \mathbb{E}_{z \sim q_{\lambda}} f
=
\mathbb{E}_{z \sim q_{\lambda}}\left[ f(z) \log q_{\lambda}(z) \right] \; .
```

The ELBO corresponds to the case where $f = \log \pi / \log q$, where $\log \pi$ is the target log density.

Instead of implementing the canonical score gradient, `ScoreGradELBO` uses the "VarGrad" objective[^RBNRA2020]:

```math
\widehat{\mathrm{VarGrad}}(\lambda) 
=
\mathrm{Var}\left( \log q_{\lambda}(z_i) - \log \pi\left(z_i\right) \right) \; ,
```

where the variance is computed over the samples $z_1, \ldots, z_m \sim q_{\lambda}$.
Differentiating the VarGrad objective corresponds to the canonical score gradient combined with the "leave-one-out" control variate[^SK2014][^KvHW2019].

[^RBNRA2020]: Richter, L., Boustati, A., Nüsken, N., Ruiz, F., & Akyildiz, O. D. (2020). Vargrad: a low-variance gradient estimator for variational inference. Advances in Neural Information Processing Systems, 33, 13481-13492.
[^SK2014]: Salimans, T., & Knowles, D. A. (2014). On using control variates with stochastic approximation for variational bayes and its connection to stochastic linear regression. arXiv preprint arXiv:1401.1022.
[^KvHW2019]: Kool, W., van Hoof, H., & Welling, M. (2019). Buy 4 reinforce samples, get a baseline for free!.
    Since the expectation of the `VarGrad` objective (not its gradient) is not exactly the ELBO, we separately obtain an unbiased estimate of the ELBO to be returned by [`estimate_objective`](@ref).
## `ScoreGradELBO`

```@docs
ScoreGradELBO
```
