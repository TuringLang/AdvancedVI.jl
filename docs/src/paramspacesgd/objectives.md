
# Overview of Algorithms

This section will provide an overview of the algorithm form by each objectives provided by `AdvancedVI`.

## Evidence Lower Bound Maximization

Evidence lower bound (ELBO) maximization[^JGJS1999] is a general family of algorithms that minimize the exclusive (or reverse) Kullback-Leibler (KL) divergence between the target distribution ``\pi`` and a variational approximation ``q_{\lambda}``.
More generally, it aims to solve the problem

```math
  \mathrm{minimize}_{q \in \mathcal{Q}}\quad \mathrm{KL}\left(q, \pi\right) \; ,
```

where $\mathcal{Q}$ is some family of distributions, often called the variational family.
Since we usually only have access to the unnormalized densities of the target distribution $\pi$, we don't have direct access to the KL divergence.
Instead, the ELBO maximization strategy maximizes a surrogate objective, the *ELBO*:

```math
  \mathrm{ELBO}\left(q\right) \triangleq \mathbb{E}_{\theta \sim q} \log \pi\left(\theta\right) + \mathbb{H}\left(q\right),
```

which is equivalent to the KL up to an additive constant (the evidence).
The ELBO and its gradient can be readily estimated through various strategies.
Overall, ELBO maximization algorithms aim to solve the problem:

```math
  \mathrm{minimize}_{q \in \mathcal{Q}}\quad -\mathrm{ELBO}\left(q\right).
```

Multiple ways to solve this problem exist, each leading to a different variational inference algorithm. `AdvancedVI` provides the following objectives:

  - [RepGradELBO](@ref repgradelbo): Implements the reparameterization gradient estimator of the ELBO gradient.
  - ScoreGradELBO: Implements the score gradient estimator of the ELBO gradient.

[^JGJS1999]: Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine learning, 37, 183-233.

