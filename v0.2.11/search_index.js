var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"vi\nADVI","category":"page"},{"location":"api/#AdvancedVI.vi","page":"API","title":"AdvancedVI.vi","text":"vi(model, alg::VariationalInference)\nvi(model, alg::VariationalInference, q::VariationalPosterior)\nvi(model, alg::VariationalInference, getq::Function, θ::AbstractArray)\n\nConstructs the variational posterior from the model and performs the optimization following the configuration of the given VariationalInference instance.\n\nArguments\n\nmodel: Turing.Model or Function z ↦ log p(x, z) where x denotes the observations\nalg: the VI algorithm used\nq: a VariationalPosterior for which it is assumed a specialized implementation of the variational objective used exists.\ngetq: function taking parameters θ as input and returns a VariationalPosterior\nθ: only required if getq is used, in which case it is the initial parameters for the variational posterior\n\n\n\n\n\n","category":"function"},{"location":"api/#AdvancedVI.ADVI","page":"API","title":"AdvancedVI.ADVI","text":"struct ADVI{AD} <: VariationalInference{AD}\n\nAutomatic Differentiation Variational Inference (ADVI) with automatic differentiation backend AD.\n\nFields\n\nsamples_per_step::Int64: Number of samples used to estimate the ELBO in each optimization step.\nmax_iters::Int64: Maximum number of gradient steps.\nadtype::Any: AD backend used for automatic differentiation.\n\n\n\n\n\n","category":"type"},{"location":"#AdvancedVI","page":"Home","title":"AdvancedVI","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"AdvancedVI provides implementations of variational Bayesian inference (VI) algorithms. VI algorithms perform scalable and computationally efficient Bayesian inference at the cost of asymptotic exactness. AdvancedVI is part of the Turing probabilistic programming ecosystem.","category":"page"}]
}