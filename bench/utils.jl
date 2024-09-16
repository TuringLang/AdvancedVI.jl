
function variational_standard_mvnormal(type::Type, n_dims::Int, family::Symbol)
    if family == :meanfield
        AdvancedVI.MeanFieldGaussian(
            zeros(type, n_dims), Diagonal(ones(type, n_dims))
        )
    else
        AdvancedVI.FullRankGaussian(
            zeros(type, n_dims), Matrix(type, I, n_dims, n_dims)
        )
    end
end

function variational_objective(objective::Symbol; kwargs...)
    if objective == :RepGradELBO
        AdvancedVI.RepGradELBO(kwargs[:n_montecarlo])
    elseif objective == :RepGradELBOSTL
        AdvancedVI.RepGradELBO(kwargs[:n_montecarlo], entropy=StickingTheLandingEntropy())
    elseif objective == :ScoreGradELBO
        AdvancedVI.ScoreGradELBO(kwargs[:n_montecarlo])
    elseif objective == :ScoreGradELBOSTL
        AdvancedVI.ScoreGradELBO(kwargs[:n_montecarlo], entropy=StickingTheLandingEntropy())
    end
end
