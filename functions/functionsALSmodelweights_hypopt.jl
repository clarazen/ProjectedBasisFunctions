function ALS_modelweights(y::Vector,maxiter,Φ::Vector{Matrix},hyp,dd::Int,tt::TT)
    # This function solves the linear system y = khr*tt with the ALS for the weight w in tensor train format.
    # INPUTS: 
    #   y       observations
    #   khr     matrices that together define a matrix with Khatri-Rao structure, as in eq. (9)
    #   rnks    ranks for the TT
    #   maxiter maximum number of iterations
    #   σₙ²     noise variance (acts as regularization parameter)
    #   dd      first core to be updated       

    ℓ²,σ_f²,σ_n²    = hyp 
    D               = order(tt)   

    # compute √Λ from hyp
    sqrtΛ           = Vector{Vector}(undef,D);
    for d = 1:D
        w           = collect(1:M[d])';
        sqrtΛ[d]    = σ_f²^(1/D)*sqrt(2π*ℓ²) .* exp.(- ℓ²/2 .* ((π.*w')./(2L[d])).^2 )
    end
    # compute √Λ*Φ
    for d = 1:D
        Φ_[d]       = Φ[d]*diagm(sqrtΛ[d])
    end
    left,right      = initsupercores(Φ_,tt);

    covdd           = Matrix{Float64}
    swipe           = [collect(dd+1:D)...,collect(D-1:-1:2)...,collect(1:dd-1)...];
    Dir             = Int.([ones(1,D-dd-1)...,-ones(1,D-1)...,ones(1,dd-1)...]);
    
    for iter = 1:maxiter
        for k = 1:2D-2
            d           = swipe[k];
            # compute product Φ*W_{\setminus d}
            if d == 1
                # left[1] is useless, only inputted to not throw error
                ΦWd     = KhRxTTm(1,left[1],right[2],Φ[1],D); 
            elseif d == D
                # right[D] is useless, only inputted to not throw error
                ΦWd     = KhRxTTm(D,left[D-1],right[D],Φ[D],D);
            else
                ΦWd     = KhRxTTm(d,left[d-1],right[d+1],Φ[d],D);
            end
            # update dth tt-core
            tmp         = ΦWd'*ΦWd + σₙ² *Matrix(I,size(ΦWd,2),size(ΦWd,2));
            tt[d]       = reshape(tmp\(ΦWd'*y),size(tt0[d])) 
            # compute residual
            res[iter,k] = norm(y - ΦWd*tt[d][:])/norm(y)
            # shift norm to next core-to-be-updated
            tt          = shiftTTnorm(tt,d,Dir[k]) 
            # compute new supercore with updated tt-core
            left,right  = getsupercores!(d,left,right,tt[d],Φ[d],Dir[k],D)              
        end
    end
    # update dd-th tt-core
    ΦWd         = KhRxTTm(dd,left[dd-1],right[dd+1],Φ[dd],D);
    tmp         = ΦWd'*ΦWd + σₙ² *Matrix(I,size(ΦWd,2),size(ΦWd,2));
    tt[dd]      = reshape(tmp\(ΦWd'*y),size(tt0[dd])) 
    covdd       = σₙ²*inv(tmp)

    return tt,covdd,res
end
