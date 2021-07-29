
## Implace implementation of gradient for ForwardDiff
function grad!(
    diff_result::DiffResults.MutableDiffResult,
    f,
    x::AbstractArray,
    alg::VariationalInference{<:ForwardDiffAD},
)
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(x), chunk_size))
    config = ForwardDiff.GradientConfig(f, x, chunk)
    return ForwardDiff.gradient!(diff_result, f, x, config)
end
