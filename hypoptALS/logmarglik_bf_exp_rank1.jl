function logmarglik_bf_exp_rank1(hyp,X,y,M)
    ℓ²,σ_f²,σ_n²    = exp.(hyp) 
    boundsMin       = minimum(X,dims=1);
    boundsMax       = maximum(X,dims=1);
    L               = ((boundsMax.-boundsMin) ./ 2)[1,:] .+ 2*ℓ²; 
    Φ,Λ,ind         = colectofbasisfunc(M,X,ℓ²,σ_f²,L);

    invΛ            = diagm(1 ./ diag(Λ));
    Z               = σ_n²*invΛ + Φ'*Φ;
    F               = svd(Z);
    Zrank1          = F.U[:,1]*F.S[1]*F.Vt[1,:]'
    invZrank1       = F.U[:,1]*(1/F.S[1])*F.Vt[1,:]'

    term1           = 1/2 * ((N-M) * log(σ_n²) + log(det(Zrank1)) + sum(log.(diag(Λ))))
    term2           = 1/2σ_n² * (y'*y - (y'*Φ) *invZrank1* (Φ'*y) )
    term3           = N/2 * log(2π)
   
    return term1 +  term2 + term3 

end
