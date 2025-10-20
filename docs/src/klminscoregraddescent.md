# [`KLMinScoreGradDescent`](@id klminscoregraddescent)

## Description

This algorithms aim to minimize the exclusive (or reverse) Kullback-Leibler (KL) divergence via stochastic gradient descent in the space of parameters.
Specifically, it uses the the *score gradient* estimator, which is similar to the algorithm that was originally referred to as black-box variational inference (BBVI; [^RGB2014][^WW2013]).
(The term BBVI has also recently been used to refer to the more general setup of maximizing the ELBO in parameter space. We are using the more narrow definition, which restricts to the use of the score gradient.)
However, instead of using the vanilla score gradient estimator, we differentiate the "VarGrad" objective[^RBNRA2020], which results in the score gradient variance-reduced by the leave-one-out control variate[^SK2014][^KvHW2019].
`KLMinScoreGradDescent` is also an alias of `BBVI`.

[^RBNRA2020]: Richter, L., Boustati, A., Nüsken, N., Ruiz, F., & Akyildiz, O. D. (2020). Vargrad: a low-variance gradient estimator for variational inference. Advances in Neural Information Processing Systems, 33, 13481-13492.
[^SK2014]: Salimans, T., & Knowles, D. A. (2014). On using control variates with stochastic approximation for variational bayes and its connection to stochastic linear regression. arXiv preprint arXiv:1401.1022.
```@docs
KLMinScoreGradDescent
```

## Methodology

This algorithm aims to solve the problem

```math
  \mathrm{minimize}_{q \in \mathcal{Q}}\quad \mathrm{KL}\left(q, \pi\right)
```

where $\mathcal{Q}$ is some family of distributions, often called the variational family, by running stochastic gradient descent in the (Euclidean) space of parameters.
That is, for all $$q_{\lambda} \in \mathcal{Q}$$, we assume $$q_{\lambda}$$ there is a corresponding vector of parameters $$\lambda \in \Lambda$$, where the space of parameters is Euclidean such that $$\Lambda \subset \mathbb{R}^p$$.

Since we usually only have access to the unnormalized densities of the target distribution $\pi$, we don't have direct access to the KL divergence.
Instead, the ELBO maximization strategy maximizes a surrogate objective, the *evidence lower bound* (ELBO; [^JGJS1999])

```math
  \mathrm{ELBO}\left(q\right) \triangleq \mathbb{E}_{\theta \sim q} \log \pi\left(\theta\right) + \mathbb{H}\left(q\right),
```

which is equivalent to the KL up to an additive constant (the evidence).

Algorithmically, `KLMinRepGradDescent` iterates the step

```math
  \lambda_{t+1} = \mathrm{operator}\big(
      \lambda_{t} + \gamma_t \widehat{\nabla_{\lambda} \mathrm{ELBO}} (q_{\lambda_t}) 
  \big) , 
```

where $\widehat{\nabla \mathrm{ELBO}}(q_{\lambda})$ is the score gradient estimate[^G1990][^KR1996][^RSU1996][^W1992] of the ELBO gradient and $$\mathrm{operator}$$ is an optional operator (*e.g.* projections, identity mapping).

Let us describe the score gradient estimator[^G1990][^KR1996][^RSU1996][^W1992] of the ELBO gradient, also known as the score function method and the REINFORCE gradient.
For variational inference, the use of the score gradient was proposed in [^WW2013][^RGB2014].
Unlike the reparameterization gradient, the score gradient does not require the target log density to be differentiable, and does not differentiate through the sampling process of the variational approximation $q$.
Instead, it only requires gradients of the log density $\log q$.
For this reason, the score gradient is the standard method to deal with discrete variables and targets with discrete support.
In more detail, the score gradient uses the Fisher log-derivative identity: For any regular $f$,

```math
\nabla_{\lambda} \mathbb{E}_{z \sim q_{\lambda}} f
=
\mathbb{E}_{z \sim q_{\lambda}}\left[ f(z) \log q_{\lambda}(z) \right] \; .
```

The ELBO corresponds to the case where $f = \log \pi / \log q$, where $\log \pi$ is the target log density.
Instead of implementing the canonical score gradient, `KLMinScoreGradDescent` internally uses the "VarGrad" objective[^RBNRA2020]:

```math
\widehat{\mathrm{VarGrad}}(\lambda) 
=
\mathrm{Var}\left( \log q_{\lambda}(z_i) - \log \pi\left(z_i\right) \right) \; ,
```

where the variance is computed over the samples $z_1, \ldots, z_m \sim q_{\lambda}$.
Differentiating the VarGrad objective corresponds to the canonical score gradient combined with the "leave-one-out" control variate[^SK2014][^KvHW2019].

[^JGJS1999]: Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine learning, 37, 183-233.
[^G1990]: Glynn, P. W. (1990). Likelihood ratio gradient estimation for stochastic systems. Communications of the ACM, 33(10), 75-84.
[^KR1996]: Kleijnen, J. P., & Rubinstein, R. Y. (1996). Optimization and sensitivity analysis of computer simulation models by the score function method. European Journal of Operational Research, 88(3), 413-427.
[^RSU1996]: Rubinstein, R. Y., Shapiro, A., & Uryasev, S. (1996). The score function method. Encyclopedia of Management Sciences, 1363-1366.
[^W1992]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8, 229-256.
[^WW2013]: Wingate, D., & Weber, T. (2013). Automated variational inference in probabilistic programming. arXiv preprint arXiv:1301.1299.
[^RGB2014]: Ranganath, R., Gerrish, S., & Blei, D. (2014). Black box variational inference. In Artificial intelligence and statistics (pp. 814-822). PMLR.
[^KvHW2019]: Kool, W., van Hoof, H., & Welling, M. (2019). Buy 4 reinforce samples, get a baseline for free!.
