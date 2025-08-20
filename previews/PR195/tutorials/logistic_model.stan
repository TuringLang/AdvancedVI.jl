data {
  int<lower=0> N;
  int<lower=0> D;
  matrix[N,D] X;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  vector[D] beta;
  real<lower=1e-5> sigma;
}
model {
  sigma ~ lognormal(0, 3);
  beta ~ normal(0, sigma);
  y ~ bernoulli_logit(X * beta);
}

