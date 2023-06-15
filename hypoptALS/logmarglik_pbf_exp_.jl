#module logmarklik

#using LinearAlgebra
#using ..functionsALSmodelweights
#using ..functionsTT

#export logmarglik_pbf_exp

function logmarglik_pbf_exp(hyp::Vector{Float64},X,y,Φ,tt,σ_y²,hyp1,hyp2)
    # used in method inspired by hyrid projection method
    push!(hyp1,exp(hyp[1]))
    push!(hyp2,exp(hyp[2]))
    ℓ²,σ_f²         = exp.(hyp);
    D               = size(Φ,1)
    boundsMin       = minimum(X,dims=1);
    boundsMax       = maximum(X,dims=1);
    L               = ((boundsMax.-boundsMin)./2 .+ 2*sqrt(ℓ²))[1,:]
    Λ               = Vector{Vector}(undef,D);
    for d = 1:D
        w           = collect(1:size(Φ[d],2))';
        Λ[d]        = σ_f²^(1/D)*2*π^(1/2)*(2ν)^ν/ℓ²^ν * (gamma(ν+1/2,0)/gamma(ν,0)) .* ((2ν/ℓ² .+ 4π^2*((π.*w')./(2L[d])).^2).^(-ν+1/2))
        #Λ[d]        = σ_f²^(1/D)*sqrt(2π*ℓ²) .* exp.(- ℓ²/2 .* ((π.*w')./(2L[d])).^2 )
    end
        
    d               = tt.normcore
    left,right      = initsupercores(Φ,tt,d);
    if d == 1
        # left[1] is useless, only inputted to not throw error
        Φp      = KhRxTTm(1,left[1],right[2],Φ[1],D); 
    elseif d == D
        # right[D] is useless, only inputted to not throw error
        Φp      = KhRxTTm(D,left[D-1],right[D],Φ[D],D);
    else
        Φp      = KhRxTTm(d,left[d-1],right[d+1],Φ[d],D);
    end

    WdΛWd           = projectedprior(tt,Λ,d);
    sqrtWdΛWd       = cholesky(Hermitian(WdΛWd)+eps(1.)*I).L;

    # compute log marginal likelihood terms
    N               = size(X,1)
    Z               = I + 1/σ_y²*sqrtWdΛWd*Φp'*Φp*sqrtWdΛWd;

    Lchol           = cholesky(Hermitian(Z+eps(1.)*I)).L;
    term1           = sum(log.(diag(Lchol))) + N/2*log(σ_y²) 
    term2           = 1/2σ_y² * y'*y - 1/2σ_y²^2 * y'*Φp*sqrtWdΛWd*(Z\(sqrtWdΛWd*Φp'*y))
    term3           = N/2 * log(2π)

    return  term1 + term2[1] + term3

end

function logmarglik_pbf_exp(logλ,y,Φ,tt,evol)
    # used in method inspired by hyrid projection method
    λ               = exp.(logλ[1]);
                    push!(evol,exp(λ))
    D               = size(Φ,1)
    N               = size(Φ[1],1)
    d               = tt.normcore
    left,right      = initsupercores(Φ,tt,d);
    Φp              = KhRxTTm(d,left[d-1],right[d+1],Φ[d],D);

    # compute log marginal likelihood terms
    Z               = Φp'*Φp + 1/λ*I;
    Lchol           = cholesky(Z).L;

    term1           = - sum(log.(diag(Lchol))) - N/2*log(λ) 
    term2           = - 1/2*y'*y + 1/2*y'*Φp*(Z\(Φp'*y))
    term3           = - N/2 * log(2π)

    return  -(term1 + term2 + term3)

end

#end