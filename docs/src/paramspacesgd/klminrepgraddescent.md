# [`KLMinRepGradDescent`](@id klminrepgraddescent)

This is a convenience constructor for [`ParamSpaceSGD`](@ref paramspacesgd) with the [`RepGradELBO`](@ref repgradelbo) objective.
This is equivalent to the algorithm commonly referred as automatic differentiation variational inference[^KTRGB2017].
`KLMinRepGradDescent` is also an alias of `ADVI` .

[^KTRGB2017]: Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. *Journal of Machine Learning Research*, 18(14), 1-45.
```@docs
KLMinRepGradDescent
```
