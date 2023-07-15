
# [Automatic Differentiation Variational Inference](@id advi)
The automatic differentiation variational inference (ADVI; Kucukelbir *et al.* 2017) objective is a method for estimating the evidence lower bound between a target posterior distribution ``\pi`` and a variational approximation ``q_{\phi,\lambda}``.
By maximizing ADVI objective, it is equivalent to solving the problem

```math
  \mathrm{minimize}_{\lambda \in \Lambda}\quad \mathrm{KL}\left(q_{\phi,\lambda}, \pi\right).
```

The key aspects of the ADVI objective are the followings:
1. The use of the reparameterization gradient estimator
2. Automatically match the support of the target posterior through "bijectors."

Thanks to Item 2, the user is free to choose any unconstrained variational family, for which
bijectors will automatically match the potentially constrained support of the target.

In particular, ADVI implicitly forms a variational approximation ``q_{\phi,\lambda}``
from a reparameterizable distribution ``q_{\lambda}`` and a bijector ``\phi`` such that
```math
z &\sim  q_{\phi,\lambda} \qquad\Leftrightarrow\qquad
z &\stackrel{d}{=} \phi^{-1}\left(\eta\right);\quad \eta \sim q_{\lambda} 
```
ADVI provides a principled way to compute the evidence lower bound for ``q_{\phi,\lambda}``.

That is,

```math
\begin{aligned}
\mathrm{ADVI}\left(\lambda\right)
&\triangleq
\mathbb{E}_{\eta \sim q_{\lambda}}\left[
  \log \pi\left( \phi^{-1}\left( \eta \right) \right)
\right]
+ \mathbb{H}\left(q_{\lambda}\right)
+ \log \lvert J_{\phi^{-1}}\left(\eta\right) \rvert \\
&=
\mathbb{E}_{\eta \sim q_{\lambda}}\left[
  \log \pi\left( \phi^{-1}\left( \eta \right) \right)
\right]
+
\mathbb{E}_{\eta \sim q_{\lambda}}\left[
  - \log q_{\lambda}\left( \eta \right) \lvert J_{\phi}\left(\eta\right) \rvert
\right] \\
&=
\mathbb{E}_{z \sim q_{\phi,\lambda}}\left[ \log \pi\left(z\right) \right]
+
\mathbb{H}\left(q_{\phi,\lambda}\right)
\end{aligned}
```

The idea of using the reparameterization gradient estimator for variational inference was first 
coined by Titsias and Lázaro-Gredilla (2014).
Bijectors were generalized by Dillon *et al.* (2017) and later implemented in Julia by
Fjelde *et al.* (2017).


```@docs
ADVI
```

# References
1. Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. Journal of machine learning research.
2. Titsias, M., & Lázaro-Gredilla, M. (2014, June). Doubly stochastic variational Bayes for non-conjugate inference. In International conference on machine learning (pp. 1971-1979). PMLR.
3. Dillon, J. V., Langmore, I., Tran, D., Brevdo, E., Vasudevan, S., Moore, D., ... & Saurous, R. A. (2017). Tensorflow distributions. arXiv preprint arXiv:1711.10604.
4. Fjelde, T. E., Xu, K., Tarek, M., Yalburgi, S., & Ge, H. (2020, February). Bijectors. jl: Flexible transformations for probability distributions. In Symposium on Advances in Approximate Bayesian Inference (pp. 1-17). PMLR.


