function logmarglik_pbf_exp(hyp,X,y,Φ,tt,σ_y²,hyp1,hyp2,costfunc)
    push!(hyp1,exp(hyp[1]))
    push!(hyp2,exp(hyp[2]))
    ℓ²,σ_f²         = exp.(hyp);
    D               = size(Φ,1)
    M               = prod(size.(Φ,2))
    L               = 2.2*ones(D)
    Λ               = Vector{Vector}(undef,D);
    for d = 1:D
        w           = collect(1:size(Φ[d],2))';
        ν           = 5/3
        #Λ[d]        = σ_f²^(1/D)*2*π^(1/2)*(2ν)^ν/ℓ²^ν * (gamma(ν+1/2,0)/gamma(ν,0)) .* ((2ν/ℓ² .+ 4π^2*((π.*w')./(2L[d])).^2).^(-ν+1/2))
        Λ[d]        = σ_f²^(1/D)*sqrt(2π*ℓ²) .* exp.(- ℓ²/2 .* ((π.*w')./(2L[d])).^2 )
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

    # find inverse with qr or svd
    WdΛWd           = projectedprior(tt,Λ,d);
    WdΛWd_chol      = cholesky(Hermitian(WdΛWd)+eps(1.)*I).L
    invWdΛWd        = inv(WdΛWd)
    Z_chol          = cholesky(Hermitian(ΦWd'*ΦWd +  σ_y²*invWdΛWd)+eps(1.)*I).L
    invZ            = inv(ΦWd'*ΦWd +  σ_y²*invWdΛWd)

    term1           = 2*sum(log.(diag(Z_chol))) + 2*sum(log.(diag(WdΛWd_chol))) + (N-M)*log(σ_y²)
    term2           = 1/σ_y²*(y'*y - y'*ΦWd*invZ*ΦWd'*y)[1]
    term3           = N*log(2π)

    push!(costfunc,(term1+term2+term3)/2)

    return (term1+term2+term3)/2
end

#=
function dlogmarglik_pbf_exp!(derivatives)
    ℓ²,σ_f²         = exp.(hyp);
    D               = size(Φ,1)
    boundsMin       = minimum(X,dims=1);
    boundsMax       = maximum(X,dims=1);
    L               = ((boundsMax.-boundsMin)./2 .+ 2*sqrt(ℓ²))[1,:]
    Λ               = Vector{Vector}(undef,D);
    for d = 1:D
        w           = collect(1:size(Φ[d],2))';
        #Λ[d]        = σ_f²^(1/D)*2*π^(1/2)*(2ν)^ν/ℓ²^ν * (gamma(ν+1/2,0)/gamma(ν,0)) .* ((2ν/ℓ² .+ 4π^2*((π.*w')./(2L[d])).^2).^(-ν+1/2))
        Λ[d]        = σ_f²^(1/D)*sqrt(2π*ℓ²) .* exp.(- ℓ²/2 .* ((π.*w')./(2L[d])).^2 )
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
    # some preprocessing to ensure that there is a cholesky factor of WdΛWd
    invWdΛWd        = inv(WdΛWd)
    invZ            = inv(ΦWd'*ΦWd +  invWdΛWd)

    ∂term1∂θ₁       = 
    ∂term2∂θ₁       = 
    ∂term1∂θ₂       = prod(size.(Φ,2)) - tr(invZ*invWdΛWd)
    ∂term2∂θ₂       = -1/σ_y² * y'*ΦWd*invZ*invWdΛWd*invZ*ΦWd'*y

    derivatives[1]  = ∂term1∂θ₁ + ∂term2∂θ₁
    derivatives[2]  = ∂term1∂θ₂ + ∂term2∂θ₂
end

=#