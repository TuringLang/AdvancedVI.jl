
function ADVI(problem, adtype, optimizer, averager)
    objective = RepGradELBO()
    return ParamSpaceSGD(problem, objective, adtype, optimizer, averager)
end

function BBVIRepGradProx(problem, adtype, optimizer, averager)
    objective = RepGradELBO()
    return ParamSpaceSGD(problem, objective, adtype, optimizer, averager)
end

function BBVIRepGradProxSTL(problem, adtype, optimizer, averager)
    objective = RepGradELBO()
    return ParamSpaceSGD(problem, objective, adtype, optimizer, averager)
end
