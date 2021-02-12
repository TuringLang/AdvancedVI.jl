function gradlogπ!(g::GradientResult, alg::VariationalInference{<:ForwardDiffAD}, logπ, q, x)
    f(x) = sum(eachcol(x)) do z
        logπ(z)
    end
    chunk_size = getchunksize(typeof(alg))
    # Set chunk size and do ForwardMode.
    chunk = ForwardDiff.Chunk(min(length(x), chunk_size))
    config = ForwardDiff.GradientConfig(f, x, chunk)
    ForwardDiff.gradient!(g, f, x, config)
end