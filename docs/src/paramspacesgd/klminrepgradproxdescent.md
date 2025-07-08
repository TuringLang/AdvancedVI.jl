
# `KLMinRepGradProxDescent`

This is a convenience constructor for [`ParamSpaceSGD`](@ref paramspacesgd) with the [`RepGradELBO`](@ref repgradelbo) objective with a proximal operator of the entropy (see [here](@ref proximalocationscaleentropy)) of location-scale variational families.
It implements the stochastic proximal gradient descent-based algorithm described in: [^D2020][^KMG2024][^DGG2023].

[^D2020]: Domke, J. (2020). Provable smoothness guarantees for black-box variational inference. In *International Conference on Machine Learning*.
[^KMG2024]: Kim, K., Ma, Y., & Gardner, J. (2024). Linear Convergence of Black-Box Variational Inference: Should We Stick the Landing?. In International Conference on Artificial Intelligence and Statistics (pp. 235-243). PMLR.
[^DGG2023]: Domke, J., Gower, R., & Garrigos, G. (2023). Provable convergence guarantees for black-box variational inference. Advances in neural information processing systems, 36, 66289-66327.

```@docs
KLMinRepGradProxDescent
```
