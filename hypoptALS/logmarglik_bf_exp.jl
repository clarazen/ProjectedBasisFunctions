function logmarglik_bf_exp_(hyp,X,y,M)
    ℓ²,σ_f²,σ_n²    = exp.(hyp) 
    boundsMin       = minimum(X,dims=1);
    boundsMax       = maximum(X,dims=1);
    L               = ((boundsMax.-boundsMin) ./ 2)[1,:] .+ 2*ℓ²; 
    Φ,Λ,ind         = colectofbasisfunc(M,X,ℓ²,σ_f²,L);

    invΛ            = diagm(1 ./ diag(Λ));
    Z               = σ_n²*invΛ + Φ'*Φ;
    invZ            = inv(Z);
    Lchol           = cholesky(Z).L;
    term1           = 1/2 * ((N-M) * log(σ_n²) + 2*sum(log.(diag(Lchol))) + sum(log.(diag(Λ))))
    term2           = 1/2σ_n² * (y'*y - y'*Φ* (Lchol'\(Lchol\(Φ'*y))) )
    term3           = N/2 * log(2π)
   
    return term1 +  term2 + term3 

end

logmarglik_bf_exp(hyp) = logmarglik_bf_exp_(hyp,X,y,M)