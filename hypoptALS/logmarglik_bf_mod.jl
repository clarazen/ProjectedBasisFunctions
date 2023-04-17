function logmarglik_bf_mod_(hyp,X,y,M)
    ℓ²,σ_f²,σ_n²    = hyp 
    boundsMin       = minimum(X,dims=1);
    boundsMax       = maximum(X,dims=1);
    L               = ((boundsMax.-boundsMin) ./ 2)[1,:] .+ 2*ℓ²; 
    Φ,Λ,ind         = colectofbasisfunc(M,X,ℓ²,σ_f²,L);

    Φ               = Φ*sqrt.(Λ)
    Z               = I + 1/σ_n²*Φ'*Φ;
    Lchol           = cholesky(Z).L;
    term1           = sum(log.(diag(Lchol))) + N/2*log(σ_n²) 
    term2           = 1/2σ_n² * y'*y - 1/2σ_n²^2 * y'*Φ*(Z\(Φ'*y))
    term3           = N/2 * log(2π)

    return  term1 + term2 + term3

end

logmarglik_bf_mod(hyp) = logmarglik_bf_mod_(hyp,X,y,M)