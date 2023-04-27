#module logmarklik

#using LinearAlgebra
#using ..functionsALSmodelweights
#using ..functionsTT

#export logmarglik_pbf_exp

function logmarglik_pbf_exp(hyp,X,y,Φ,tt,hyp1,hyp2,hyp3)
    # used in method inspired by hyrid projection method
    push!(hyp1,exp(hyp[1]))
    push!(hyp2,exp(hyp[2]))
    push!(hyp3,exp(hyp[3]))
    ℓ²,σ_f²,σ_n²    = exp.(hyp);
    D               = size(Φ,1)
    boundsMin       = minimum(X,dims=1);
    boundsMax       = maximum(X,dims=1);
    L               = ((boundsMax.-boundsMin)./2 .+ 2*sqrt(ℓ²))[1,:]
    sqrtΛ           = Vector{Vector}(undef,D);
    for d = 1:D
        w           = collect(1:size(Φ[d],2))';
        sqrtΛ[d]    = sqrt.(σ_f²^(1/D)*sqrt(2π*ℓ²) .* exp.(- ℓ²/2 .* ((π.*w')./(2L[d])).^2 ))
    end
    # compute √Λ*Φ
    Φ_              = Vector{Matrix}(undef,D)
    for d = 1:D
        Φ_[d]       = Φ[d]*diagm(sqrtΛ[d])
    end
    
    d               = tt.normcore
    left,right      = initsupercores(Φ_,tt,d);
    if d == 1
        # left[1] is useless, only inputted to not throw error
        Φp      = KhRxTTm(1,left[1],right[2],Φ[1],D); 
    elseif d == D
        # right[D] is useless, only inputted to not throw error
        Φp      = KhRxTTm(D,left[D-1],right[D],Φ[D],D);
    else
        Φp      = KhRxTTm(d,left[d-1],right[d+1],Φ[d],D);
    end
    # compute log marginal likelihood terms
    N               = size(X,1)
    Z               = I + 1/σ_n²*Φp'*Φp;
    Lchol           = cholesky(Z).L;
    term1           = sum(log.(diag(Lchol))) + N/2*log(σ_n²) 
    term2           = 1/2σ_n² * y'*y - 1/2σ_n²^2 * y'*Φp*(Z\(Φp'*y))
    term3           = N/2 * log(2π)

    return  term1 + term2 + term3

end



#end