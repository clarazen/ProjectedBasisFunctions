function logmarglik_pbf_(hyp,X,y,M,Wd)
    ℓ²,σ_f²,σ_n²    = hyp 
    boundsMin       = minimum(X,dims=1);
    boundsMax       = maximum(X,dims=1);
    L               = ((boundsMax.-boundsMin) ./ 2)[1,:] .+ 2*ℓ²; 
    Φ               = colectofbasisfunc(M,X,ℓ²,σ_f²,L);

    Φp              = khrtimesttm(Φ,Wd);
    Z               = I + 1/σ_n²*Φp'*Φp;
    Lchol           = cholesky(Z).L;
    term1           = sum(log.(diag(Lchol))) + N/2*log(σ_n²) 
    term2           = 1/2σ_n² * y'*y - 1/2σ_n²^2 * y'*Φp*(Z\(Φp'*y))
    term3           = N/2 * log(2π)

    return  term1 + term2 + term3

end

logmarglik_pbf(hyp) = logmarglik_pbf_(hyp,X,y,M,Wd)




