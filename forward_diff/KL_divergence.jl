using Base.Test
using ForwardDiff


#######################################
# Multivariate normal

function diagmvn_mvn_kl{NumType <: Number, NumType2 <: Number}(
  mean1::Vector{NumType2}, vars1::Vector{NumType2}, mean2::Vector{NumType}, cov2::Matrix{NumType}, calculate_derivs::Bool)
    const precision2 = cov2^-1
    const logdet_cov2 = logdet(cov2)
    const K = length(mean2)

    diff = mean2 - mean1

    kl = sum(diag(precision2) .* vars1) - K
    kl += (diff' * precision2 * diff)[]
    kl += -sum(log(vars1)) + logdet_cov2
    kl = 0.5 * kl

    if calculate_derivs
        grad_mean = zeros(NumType2, K)
        grad_var = zeros(NumType2, K)
        hess_mean = zeros(NumType2, K, K)
        hess_var = zeros(NumType2, K, K)

        grad_mean = -1 * precision2 * diff
        grad_var = 0.5 * (diag(precision2) - 1 ./ var1)

        hess_mean = precision2
        for k in 1:K
            hess_var[k, k] = 0.5 ./ (var1[k] ^ 2)
        end

        return kl, grad_mean, grad_var, hess_mean, hess_var
    else
        return kl
    end
end


K = 4
mean1 = rand(K)
var1 = rand(K)
var1 = var1 .* var1

mean2 = rand(K)
cov2 = rand(K, K)
cov2 = 0.2 * cov2 * cov2' + eye(K)

kl, grad_mean, grad_var, hess_mean, hess_var = diagmvn_mvn_kl(mean1, var1, mean2, cov2, true);

hess = zeros(Float64, 2 * K, 2 * K)
hess[1:K, 1:K] = hess_mean
hess[(K + 1):(2 * K), (K + 1):(2 * K)] = hess_var

function diagmvn_mvn_kl_wrapper{NumType <: Number}(par::Vector{NumType})
    K = Int(length(par) / 2)
    mean1 = par[1:K]
    var1 = par[(K + 1):(2 * K)]
    diagmvn_mvn_kl(mean1, var1, mean2, cov2, false)
end

par = vcat(mean1, var1)
ad_grad = ForwardDiff.gradient(diagmvn_mvn_kl_wrapper, par)
ad_hess = ForwardDiff.hessian(diagmvn_mvn_kl_wrapper, par)

@test_approx_eq diagmvn_mvn_kl_wrapper(par) kl
@test_approx_eq ad_grad vcat(grad_mean, grad_var)
@test_approx_eq ad_hess hess

#######################################
# Categorical

function categorical_kl{NumType <: Number, NumType2 <: Number}(p1::Vector{NumType2}, p2::Vector{NumType}, calculate_derivs::Bool)
    kl = zero(NumType2)

    if calculate_derivs
        grad = zeros(NumType2, length(p1))
        hess = zeros(NumType2, length(p1), length(p1))
    end

    for i in 1:length(p1)
        log_ratio = log(p1[i]) - log(p2[i])
        kl += p1[i] * log_ratio
        if calculate_derivs
            grad[i] = 1 + log_ratio
            hess[i, i] = 1 / p1[i]
        end
    end

    if calculate_derivs
        return kl, grad, hess
    else
        return kl
    end
end


k = 4
p1 = rand(k)
p2 = rand(k)

p1 = p1 / sum(p1)
p2 = p2 / sum(p2)

kl, grad, hess = categorical_kl(p1, p2, true);


function categorical_kl_wrapper{NumType <: Number}(par::Vector{NumType})
    return categorical_kl(par, p2, false)
end

ad_grad = ForwardDiff.gradient(categorical_kl_wrapper, p1)
ad_hess = ForwardDiff.hessian(categorical_kl_wrapper, p1)

@test_approx_eq categorical_kl_wrapper(p1) kl
@test_approx_eq ad_grad grad
@test_approx_eq ad_hess hess


#######################################
# Beta

function beta_kl{NumType <: Number, NumType2 <: Number}(alpha1::NumType2, beta1::NumType2, alpha2::NumType, beta2::NumType, calculate_derivs::Bool)
    alpha_diff = alpha1 - alpha2
    beta_diff = beta1 - beta2
    both_inv_diff = -(alpha_diff + beta_diff)
    di_both1 = digamma(alpha1 + beta1)

    log_term = lgamma(alpha1 + beta1) - lgamma(alpha1) - lgamma(beta1)
    log_term -= lgamma(alpha2 + beta2) - lgamma(alpha2) - lgamma(beta2)
    apart_term = alpha_diff * digamma(alpha1) + beta_diff * digamma(beta1)
    together_term = both_inv_diff * di_both1
    kl = log_term + apart_term + together_term

    if calculate_derivs
        grad = zeros(NumType2, 2)
        hess = zeros(NumType2, 2, 2)

        trigamma_alpha1 = trigamma(alpha1)
        trigamma_beta1 = trigamma(beta1)
        trigamma_both = trigamma(alpha1 + beta1)
        grad[1] = alpha_diff * trigamma_alpha1 + both_inv_diff * trigamma_both
        grad[2] = beta_diff * trigamma_beta1 + both_inv_diff * trigamma_both

        quadgamma_both = polygamma(2, alpha1 + beta1)
        hess[1, 1] = alpha_diff * polygamma(2, alpha1) + both_inv_diff * quadgamma_both +
                     trigamma_alpha1 - trigamma_both
        hess[2, 2] = beta_diff * polygamma(2, beta1) + both_inv_diff * quadgamma_both +
                     trigamma_beta1 - trigamma_both
        hess[1, 2] = hess[2, 1] = -trigamma_both + both_inv_diff * quadgamma_both

        return kl, grad, hess
    else
        return kl
    end
end


alpha2 = 3.5
beta2 = 4.3

alpha1 = 4.1
beta1 = 3.9

par = Float64[ alpha1, beta1 ]
kl, grad, hess = beta_kl(alpha1, beta1, alpha2, beta2, true)

function beta_kl_wrapper{NumType <: Number}(par::Vector{NumType})
    alpha1 = par[1]
    beta1 = par[2]
    return beta_kl(alpha1, beta1, alpha2, beta2, false)
end

ad_grad = ForwardDiff.gradient(beta_kl_wrapper, par)
ad_hess = ForwardDiff.hessian(beta_kl_wrapper, par)

@test_approx_eq beta_kl_wrapper(par) kl
@test_approx_eq ad_grad grad
@test_approx_eq ad_hess hess
