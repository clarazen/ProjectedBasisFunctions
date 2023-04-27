function logmarglik_full_exp(hyp,X,y)
    ℓ²,σ_f²,σ_n²    = exp.(hyp) 
    N               = size(X,1);

    K               = covSE(X,X,[ℓ²,σ_f²,σ_n²])
    Ky              = K + σ_n²*I
    Lchol           = cholesky(Ky).L
    N               = size(X,1);

    α               = Lchol'\(Lchol\y)

    term1           = sum(log.(diag(Lchol)))
    term2           = 1/2 * y'*α
    term3           = N/2 * log(2π)

    return term1 +  term2 + term3 

end
