function ∂full(X,hyp,opt)
    ℓ²,σ_f²,σ_n²    = hyp
    K               = covSE(X,X,hyp)
    Lchol           = cholesky(K+σ_n²*I).L;
    α               = Lchol'\(Lchol\y)
    logmarglik      = sum(log.(diag(Lchol))) + 1/2 * y'*α + N/2 * log(2π)    

    if opt == 1
        inner           = reshape([norm(X[i,:]-X[j,:])^2/(2ℓ²^2) for i = 1:N for j = 1:N],N,N);
        ∂θ₁_1           = tr(inv(K+σ_n²*I)*(K.*inner))
        ∂θ₂_1           = tr(inv(K+σ_n²*I)*K/σ_f²)
        ∂θ₃_1           = tr(inv(K+σ_n²*I))
        ∂θ₁_2           = α'*(K.*inner)*α
        ∂θ₂_2           = α'*K/σ_f²*α
        ∂θ₃_2           = α'*α

        return 1/2*(∂θ₁_1 - ∂θ₁_2), 1/2*(∂θ₂_1 - ∂θ₂_2), 1/2*(∂θ₃_1 - ∂θ₃_2)
    else
        inner           = reshape([norm(X[i,:]-X[j,:])^2/2ℓ² for i = 1:N for j = 1:N],N,N);
        ∂θ₁_1           = tr(inv(K+σ_n²*I)*(K.*inner))
        ∂θ₂_1           = tr(inv(K+σ_n²*I)*K)
        ∂θ₃_1           = tr(inv(K+σ_n²*I)*(σ_n²*I))
        ∂θ₁_2           = α'*(K.*inner)*α
        ∂θ₂_2           = α'*K*α
        ∂θ₃_2           = α'*(σ_n²*I)*α

        return 1/2*(∂θ₁_1 - ∂θ₁_2), 1/2*(∂θ₂_1 - ∂θ₂_2), 1/2*(∂θ₃_1 - ∂θ₃_2)
    end

end

function ∂bf(X,hyp,M,opt)
    ℓ²,σ_f²,σ_n²    = hyp
    boundsMin       = minimum(X,dims=1);
    boundsMax       = maximum(X,dims=1);
    L               = ((boundsMax.-boundsMin) ./ 2)[1,:] .+ 2*hyp[1]; 
    Φ,Λ,ind         = colectofbasisfunc(M,X,hyp[1],hyp[2],L);

    invΛ            = diagm(1 ./ diag(Λ));
    Z               = σ_n²*invΛ + Φ'*Φ;
    invZ            = inv(Z);
    
    if opt == 1
        λ               = sum([((π*ind[d,:])/(2L[d])).^2 for d = 1:D]);
        ∂θ₁_1           = sum(-λ/2 .+ 1/2ℓ²) - σ_n² * tr(invZ* (invΛ .* diagm(-λ/2 .+ 1/2ℓ²)) )
        ∂θ₂_1           = M/σ_f² - σ_n² * tr(invZ*(invΛ*(1/σ_f²*I)))
        ∂θ₃_1           = (N-M)/σ_n² + tr(invZ*invΛ)

        ∂θ₁_2           = -y'*Φ*invZ*(invΛ.* diagm(-λ/2 .+ 1/2ℓ²))*invZ*Φ'*y
        ∂θ₂_2           = -y'*Φ*invZ*invΛ*(1/σ_f²*I)*invZ*Φ'*y
        ∂θ₃_2           = -2(y'*Φ*invZ*invΛ*invZ*Φ'*y)/σ_n² + (-y'*y - y'*Φ*invZ*Φ'*Φ*invZ*Φ'*y)/σ_n²^2 

        return 1/2*(∂θ₁_1 + ∂θ₁_2), 1/2*(∂θ₂_1 + ∂θ₂_2), 1/2*(∂θ₃_1 + ∂θ₃_2)
    else # with exponents
        λ               = sum([((π*ind[d,:])/(2L[d])).^2 for d = 1:D]);
        ∂θ₁_1           = sum(-λ/2*ℓ² .+ 1/2) - σ_n² * tr(invZ* (invΛ .* diagm(-λ/2*ℓ² .+ 1/2)) )
        ∂θ₂_1           = M - σ_n² * tr(invZ*invΛ)
        ∂θ₃_1           = (N-M) + tr(invZ*(invΛ*σ_n²))

        ∂θ₁_2           = -y'*Φ*invZ*(invΛ.* diagm(-λ/2*ℓ² .+ 1/2))*invZ*Φ'*y
        ∂θ₂_2           = -y'*Φ*invZ*invΛ*invZ*Φ'*y
        ∂θ₃_2           = 1/σ_n² * (-y'*y + y'*Φ*invZ*(Φ'*Φ+2σ_n²*invΛ)*invZ*Φ'*y )

        return 1/2*(∂θ₁_1 + ∂θ₁_2), 1/2*(∂θ₂_1 + ∂θ₂_2), 1/2*(∂θ₃_1 + ∂θ₃_2)
    end

end


#=
function ∂bfmod(opt)
    # modified

    if opt == 1

        return 1/2*(∂θ₁_1 - ∂θ₁_2), 1/2*(∂θ₂_1 - ∂θ₂_2), 1/2*(∂θ₃_1 - ∂θ₃_2)
    else # with exponents
        
        
        return 1/2*(∂θ₁_1 - ∂θ₁_2), 1/2*(∂θ₂_1 - ∂θ₂_2), 1/2*(∂θ₃_1 - ∂θ₃_2)
    end

end

