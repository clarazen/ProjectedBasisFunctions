function logmarglik_full_(hyp,X,y)
    ℓ²,σ_f²,σ_n²    = hyp
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

logmarglik_full(hyp) = logmarglik_full_(hyp,X,y)

