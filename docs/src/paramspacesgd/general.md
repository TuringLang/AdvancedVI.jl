
# [General](@id paramspacesgd)

`ParamSpaceSGD` SGD is a general algorithm for leveraging automatic differentiation and SGD.
Furthermore, it operates in the space of *variational parameters*.
Consider the case where each member $q_{\lambda} \in \mathcal{Q}$ of the variational family $\mathcal{Q}$ is uniquely represented through a collection of parameters $\lambda \in \Lambda \subseteq \mathbb{R}^p$. 
That is,

```math
\mathcal{Q} = \{q_{\lambda} \mid \lambda \in \Lambda \},
```
Then, as implied by the name, `ParamSpaceSGD` runs SGD on $\Lambda$, the (Euclidean) space of parameters.

Any algorithm that operates by iterating the following steps can easily be implemented via  `ParamSpaceSGD`:

1. Obtain an unbiased estimate of the target objective.
2. Obtain an estimate of the gradient of the objective by differentiating the objective estimate with respect to the parameters.
3. Perform gradient descent with the stochastic gradient estimate.

After some simplifications, each `step` of `ParamSpaceSGD` can be described as follows:

```julia
function step(rng, alg::ParamSpaceSGD, state, callback, objargs...; kwargs...)
    (; adtype, problem, objective, operator, averager) = alg
    (; q, iteration, grad_buf, opt_st, obj_st, avg_st) = state
    iteration += 1

    # Extract variational parameters of `q`
    params, re = Optimisers.destructure(q)

    # Estimate gradient and update the `DiffResults` buffer `grad_buf`.
    grad_buf, obj_st, info = estimate_gradient!(...)

    # Gradient descent step.
    grad = DiffResults.gradient(grad_buf)
    opt_st, params = Optimisers.update!(opt_st, params, grad)

    # Apply operator
    params = apply(operator, typeof(q), opt_st, params, re)

    # Apply parameter averaging
    avg_st = apply(averager, avg_st, params)

    # Updated state
    state = ParamSpaceSGDState(re(params), iteration, grad_buf, opt_st, obj_st, avg_st)
    state, false, info
end
```
The output of `ParamSpaceSGD` is the final state of `averager`.
Furthermore, `operator` can be anything from an identity mapping, a projection operator, a proximal operator, and so on.

## `ParamSpaceSGD`
The constructor for `ParamSpaceSGD` is as follows:

```@docs
ParamSpaceSGD
```

## Objective Interface

To define an instance of a `ParamSpaceSGD` algorithm, it suffices to implement the `AbstractVariationalObjective` interface.
First, we need to define a subtype of `AbstractVariationalObjective`:

```@docs
AdvancedVI.AbstractVariationalObjective
```

In addition, we need to implement some methods associated with the objective.
First, each objective may maintain a state such as buffers, online estimates of control variates, batch iterators for subsampling, and so on.
Such things should be initialized by implementing the following:

```@docs
AdvancedVI.init(
    ::Random.AbstractRNG,
    ::AdvancedVI.AbstractVariationalObjective,
    ::ADTypes.AbstractADType,
    ::Any,
    ::Any,
    ::Any,
    ::Any,
)
```
If this method is not implemented, the state will be automatically be `nothing`.

Next, the key functionality of estimating stochastic gradients should be implemented through the following:

```@docs
AdvancedVI.estimate_gradient!
```

`AdvancedVI` only interacts with each variational objective by querying gradient estimates.
In a lot of cases, however, it is convinient to be able to estimate the current value of the objective.
For example, for monitoring convergence.
This should be done through the following:

```@docs
AdvancedVI.estimate_objective
```
