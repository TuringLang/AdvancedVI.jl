# [`KLMinSqrtNaturalGradDescent`](@id klminsqrtnaturalgraddescent)

## Description

This algorithm aims to minimize the exclusive (or reverse) Kullback-Leibler (KL) divergence by running natural gradient descent.
`KLMinSqrtNaturalGradDescent` is a specific implementation of natural gradient variational inference (NGVI) also known as square-root variational Newton[^KMKL2025][^LDEBTM2024][^LDLNKS2023][^T2025].
This algorithm operates under the square-root or Cholesky factorization of the covariance matrix parameterization.
This contrasts with [`KLMinNaturalGradDescent`](@ref klminnaturalgraddescent), which operates in the precision matrix parameterization, requiring a matrix inverse at each step.
As a result, the cost of `KLMinSqrtNaturalGradDescent` should be relatively cheaper.
Since `KLMinSqrtNaturalGradDescent` is a measure-space algorithm, its use is restricted to full-rank Gaussian variational families (`FullRankGaussian`) that make the updates tractable.

```@docs
KLMinSqrtNaturalGradDescent
```

The associated objective value can be estimated through the following:

```@docs; canonical=false
estimate_objective(
    ::Random.AbstractRNG,
    ::KLMinWassFwdBwd,
    ::MvLocationScale,
    ::Any;
    ::Int,
)
```

[^KMKL2025]: Kumar, N., Möllenhoff, T., Khan, M. E., & Lucchi, A. (2025). Optimization Guarantees for Square-Root Natural-Gradient Variational Inference. *Transactions of Machine Learning Research*.
[^LDEBTM2024]: Lin, W., Dangel, F., Eschenhagen, R., Bae, J., Turner, R. E., & Makhzani, A. (2024). Can We Remove the Square-Root in Adaptive Gradient Methods? A Second-Order Perspective. In *International Conference on Machine Learning*.
[^LDLNKS2023]: Lin, W., Duruisseaux, V., Leok, M., Nielsen, F., Khan, M. E., & Schmidt, M. (2023). Simplifying momentum-based positive-definite submanifold optimization with applications to deep learning. In *International Conference on Machine Learning*.
[^T2025]: Tan, L. S. (2025). Analytic natural gradient updates for Cholesky factor in Gaussian variational approximation. *Journal of the Royal Statistical Society: Series B.*
## [Methodology](@id klminsqrtnaturalgraddescent_method)

This algorithm aims to solve the problem

```math
  \mathrm{minimize}_{q_{\lambda} \in \mathcal{Q}}\quad \mathrm{KL}\left(q_{\lambda}, \pi\right)
```

where $\mathcal{Q}$ is some family of distributions, often called the variational family, by running stochastic gradient descent in the (Euclidean) space of parameters.
That is, for all $$q_{\lambda} \in \mathcal{Q}$$, we assume $$q_{\lambda}$$ there is a corresponding vector of parameters $$\lambda \in \Lambda$$, where the space of parameters is Euclidean such that $$\Lambda \subset \mathbb{R}^p$$.

Since we usually only have access to the unnormalized densities of the target distribution $\pi$, we don't have direct access to the KL divergence.
Instead, the ELBO maximization strategy minimizes a surrogate objective, the *negative evidence lower bound*[^JGJS1999]

```math
  \mathcal{L}\left(q\right) \triangleq \mathbb{E}_{\theta \sim q} -\log \pi\left(\theta\right) - \mathbb{H}\left(q\right),
```

which is equivalent to the KL up to an additive constant (the evidence).

While `KLMinSqrtNaturalGradDescent` is close to a natural gradient variational inference algorithm, it can be derived in a variety of different ways.
In fact, the update rule has been concurrently developed by several research groups[^KMKL2025][^LDEBTM2024][^LDLNKS2023][^T2025].
Here, we will present the derivation by Kumar *et al.* [^KMKL2025].
Consider the ideal natural gradient descent algorithm discussed [here](@ref klminnaturalgraddescent_method).
This can be viewed as a discretization of the continuous-time dynamics given by the differential equation

```math
\dot{\lambda}_t
=
{F(\lambda)}^{-1} \nabla_{\lambda} \mathcal{L}\left(q_{\lambda}\right) .
```

This is also known as the *natural gradient flow*.
Notice that the flow is over the parameters $\lambda_t$.
Therefore, the natural gradient flow depends on the way we parametrize $q_{\lambda}$.
For Gaussian variational families, if we specifically choose the *square-root* (or Cholesky) parametrization such that $q_{\lambda_t} = \mathrm{Normal}(m_t, C_t C_t)$, the flow of $\lambda_t = (m_t, C_t)$ given as

```math
\begin{align*}
\dot{m}_t &= C_t C_t^{\top} \mathbb{E}_{q_{\lambda_t}} \left[ \nabla \log \pi \right] 
\\
\dot{C}_t &= C_t M\left( \mathrm{I}_d + C_t^{\top} \mathbb{E}\left[ \nabla^2 \log \pi \right] C_t \right) ,
\end{align*}  
```

where $M$ is a $\mathrm{tril}$-like function defined as

```math
{[ M(A) ]}_{ij} = \begin{cases}
    0 & \text{if $i > j$} \\
    \frac{1}{2} A_{ii} & \text{if $i = j$} \\
    A_{ij} & \text{if $i < j$} .
\end{cases}
```

`KLMinSqrtNaturalGradDescent` corresponds to the forward Euler discretization of this flow.

[^JGJS1999]: Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine learning, 37, 183-233.
