
"""
    ParamSpaceSGD(
        problem,
        objective::AbstractVariationalObjective,
        adtype::ADTypes.AbstractADType,
        optimizer::Optimisers.AbstractRule,
        averager::AbstractAverager,
        operator::AbstractOperator,
    )

This algorithm applies stochastic gradient descent (SGD) to the variational `objective` over the (Euclidean) space of variational parameters.

The trainable parameters in the variational approximation are expected to be extractable through `Optimisers.destructure`.
This requires the variational approximation to be marked as a functor through `Functors.@functor`.

# Arguments
- `objective`: Variational Objective.
- `adtype`: Automatic differentiation backend. 
- `optimizer`: Optimizer used for inference.
- `averager` : Parameter averaging strategy.
- `operator` : Operator applied to the parameters after each optimization step.

# Output
- `q_averaged`: The variational approximation formed from the averaged SGD iterates.

# Callback
The callback function `callback` has a signature of

    callback(; rng, iteration, restructure, params, averaged_params, restructure, gradient)

The arguments are as follows:
- `stat`: Statistics gathered during the current iteration. The content will vary depending on `objective`.
- `state`: Collection of the internal states used for optimization.
- `params`: Variational parameters.
- `averaged_params`: Variational parameters averaged according to the averaging strategy.
- `restructure`: Function that restructures the variational approximation from the variational parameters. Calling `restructure(param)` reconstructs the variational approximation. 
- `gradient`: The estimated (possibly stochastic) gradient.

"""
struct ParamSpaceSGD{
    Prob,
    Obj<:AbstractVariationalObjective,
    AD<:ADTypes.AbstractADType,
    Opt<:Optimisers.AbstractRule,
    Avg<:AbstractAverager,
    Op<:AbstractOperator,
} <: AbstractAlgorithm
    problem::Prob
    objective::Obj
    adtype::AD
    optimizer::Opt
    averager::Avg
    operator::Op
end

struct ParamSpaceSGDState{Q,GradBuf,OptSt,ObjSt,AvgSt}
    q::Q
    iteration::Int
    grad_buf::GradBuf
    opt_st::OptSt
    obj_st::ObjSt
    avg_st::AvgSt
end

function init(rng::Random.AbstractRNG, alg::ParamSpaceSGD, q_init)
    (; problem, adtype, optimizer, averager, objective) = alg
    params, re = Optimisers.destructure(q_init)
    opt_st = Optimisers.setup(optimizer, params)
    obj_st = init(rng, objective, adtype, problem, params, re)
    avg_st = init(averager, params)
    grad_buf = DiffResults.DiffResult(zero(eltype(params)), similar(params))
    return ParamSpaceSGDState(q_init, 0, grad_buf, opt_st, obj_st, avg_st)
end

function output(alg::ParamSpaceSGD, state)
    params_avg = value(alg.averager, state.avg_st)
    _, re = Optimisers.destructure(state.q)
    return re(params_avg)
end

function step(
    rng::Random.AbstractRNG, alg::ParamSpaceSGD, state, callback, objargs...; kwargs...
)
    (; adtype, problem, objective, operator, averager) = alg
    (; q, iteration, grad_buf, opt_st, obj_st, avg_st) = state

    iteration += 1

    params, re = Optimisers.destructure(q)

    grad_buf, obj_st, info = estimate_gradient!(
        rng, objective, adtype, grad_buf, problem, params, re, obj_st, objargs...
    )

    grad = DiffResults.gradient(grad_buf)
    opt_st, params = Optimisers.update!(opt_st, params, grad)
    params = apply(operator, typeof(q), opt_st, params, re)
    avg_st = apply(averager, avg_st, params)

    state = ParamSpaceSGDState(re(params), iteration, grad_buf, opt_st, obj_st, avg_st)

    if !isnothing(callback)
        averaged_params = value(averager, avg_st)
        info′ = callback(;
            rng,
            iteration,
            restructure=re,
            params=params,
            averaged_params=averaged_params,
            gradient=grad,
            state=state,
        )
        info = !isnothing(info′) ? merge(info′, info) : info
    end
    state, false, info
end
