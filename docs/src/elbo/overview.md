
# [Evidence Lower Bound Maximization](@id elbomax)
## Introduction

Evidence lower bound (ELBO) maximization[^JGJS1999] is a general family of algorithms that minimize the exclusive (or reverse) Kullback-Leibler (KL) divergence between the target distribution ``\pi`` and a variational approximation ``q_{\lambda}``.
More generally, they aim to solve the following problem:

```math
  \mathrm{minimize}_{q \in \mathcal{Q}}\quad \mathrm{KL}\left(q, \pi\right),
```
where $$\mathcal{Q}$$ is some family of distributions, often called the variational family.
Since the target distribution ``\pi`` is intractable in general, the KL divergence is also intractable.
Instead, the ELBO maximization strategy maximizes a surrogate objective, the *ELBO*, which serves as a lower bound to the KL. ELBO is defined as
```math
  \mathrm{ELBO}\left(q\right) \triangleq \mathbb{E}_{\theta \sim q} \log \pi\left(\theta\right) + \mathbb{H}\left(q\right),
```
which can be readily estimated through various strategies.
Overall, we solve the problem 
```math
  \mathrm{maximize}_{q \in \mathcal{Q}}\quad \mathrm{ELBO}\left(q\right).
```

Multiple ways to solve this problem exist, leading to different variational inference algorithms.

## Algorithms
Currently, `AdvancedVI` only provides the approach known as black-box variational inference (also known as Monte Carlo VI, Stochastic Gradient VI).
(Introduced independently by two groups [^RGB2014][^TL2014] in 2014.)
In particular, `AdvancedVI` focuses on the reparameterization gradient estimator[^TL2014][^RMW2014][^KW2014], which is generally superior compared to alternative strategies[^XQKS2019], discussed in the following section:
* [RepGradELBO](@ref repgradelbo)

[^JGJS1999]: Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine learning, 37, 183-233.
[^TL2014]: Titsias, M., & Lázaro-Gredilla, M. (2014). Doubly stochastic variational Bayes for non-conjugate inference. In *International Conference on Machine Learning*. 
[^RMW2014]: Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. In *International Conference on Machine Learning*.
[^KW2014]: Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. In *International Conference on Learning Representations*.
[^XQKS2019]: Xu, M., Quiroz, M., Kohn, R., & Sisson, S. A. (2019). Variance reduction properties of the reparameterization trick. In *The International Conference on Artificial Intelligence and Statistics.
[^RGB2014]: Ranganath, R., Gerrish, S., & Blei, D. (2014). Black box variational inference. In *Artificial Intelligence and Statistics*.
