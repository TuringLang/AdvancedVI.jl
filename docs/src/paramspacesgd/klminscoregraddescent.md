# [`KLMinScoreGradDescent`](@id klminscoregraddescent)

This is a convenience constructor for [`ParamSpaceSGD`](@ref paramspacesgd) with the [`ScoreGradELBO`](@ref scoregradelbo) objective.
This is similar to the algorithm that was originally referred to as black-box variational inference (BBVI; [^RGB2014][^WW2013]).
(The term BBVI has also recently been used to refer to the more general setup of maximizing the ELBO in parameter space. We are using the more narrow definition, which restricts to the use of the score gradient.)
However, instead of using the vanilla score gradient estimator, we differentiate the "VarGrad" objective[^RBNRA2020], which results in the score gradient variance-reduced by the leave-one-out control variate[^SK2014][^KvHW2019].

[^RGB2014]: Ranganath, R., Gerrish, S., & Blei, D. (2014, April). Black box variational inference. In *Artificial Intelligence and Statistics* (pp. 814-822). PMLR.
[^WW2013]: Wingate, D., & Weber, T. (2013). Automated variational inference in probabilistic programming. arXiv preprint arXiv:1301.1299.
[^RBNRA2020]: Richter, L., Boustati, A., Nüsken, N., Ruiz, F., & Akyildiz, O. D. (2020). Vargrad: a low-variance gradient estimator for variational inference. Advances in Neural Information Processing Systems, 33, 13481-13492.
[^SK2014]: Salimans, T., & Knowles, D. A. (2014). On using control variates with stochastic approximation for variational bayes and its connection to stochastic linear regression. arXiv preprint arXiv:1401.1022.
[^KvHW2019]: Kool, W., van Hoof, H., & Welling, M. (2019). Buy 4 reinforce samples, get a baseline for free!.
```@docs
KLMinScoreGradDescent
```
