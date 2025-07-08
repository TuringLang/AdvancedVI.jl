
# `KLMinScoreGradDescent`

This is a convenience constructor for [`ParamSpaceSGD`](@ref paramspacesgd) with the [`ScoreGradELBO`](@ref scoregradelbo) objective.
This is similar to the algorithm that was originally referred to as black-box variational inference (BBVI; [^RGB2014]).
(The term BBVI has also recently been used to refer to the more general setup of maximizing the ELBO in parameter space. We are using the more narrow definition, which restricts to the use of the score gradient.)
However, instead of using the vanilla score gradient estimator, we use the variance-reduced variant using the leave-one-out control variate, which is also known as the gradient of the "VarGrad" objective.

[^RGB2014]: Ranganath, R., Gerrish, S., & Blei, D. (2014, April). Black box variational inference. In *Artificial Intelligence and Statistics* (pp. 814-822). PMLR.

```@docs
KLMinScoreGradDescent
```
