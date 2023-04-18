module functionsALSmodelweights

using LinearAlgebra
using SparseArrays
using ..functionsTT
using ..functions_KhatriRao_Kronecker

export ALS_modelweights,initsupercores,KhRxTTm

function ALS_modelweights(y::Vector,khr::Vector{Matrix},rnks::Vector{Int},maxiter,σₙ²::Float64)
# This function solves the linear system y = khr*tt with the ALS for the weight w in tensor train format.
# INPUTS: 
#   y       observations
#   khr     matrices that together define a matrix with Khatri-Rao structure, as in eq. (9)
#   rnks    ranks for the TT
#   maxiter maximum number of iterations
#   σₙ²     noise variance (acts as regularization parameter)

    D           = size(khr,1)
    Md          = size(khr[1],2)

##########################################################################
    # create site-D canonical initial tensor train    
    cores = Vector{Array{Float64,3}}(undef,D);
    for i = 1:D-1 
        tmp = qr(rand(rnks[i]*Md, rnks[i+1]));
        cores[i] = reshape(Matrix(tmp.Q),(rnks[i], Md, rnks[i+1]));
    end
    cores[D]    = reshape(rand(rnks[D]*Md),(rnks[D], Md, 1));
    tt0         = TT(cores,D);
###########################################################################
    # initialize 
    tt          = tt0
    res         = zeros(maxiter,2D-2)
    left,right  = initsupercores(khr,tt0);
    swipe       = [collect(D:-1:2)..., collect(1:D-1)...];
    Dir         = Int.([-ones(1,D-1)...,ones(1,D-1)...]);
    for iter = 1:maxiter
        for k = 1:2D-2
            d           = swipe[k];
            # compute product Φ*W_{\setminus d}
            if d == 1
                # left[1] is useless, only inputted to not throw error
                ΦWd     = KhRxTTm(d,left[1],right[2],khr[1],D); 
            elseif d == D
                # right[D] is useless, only inputted to not throw error
                ΦWd     = KhRxTTm(D,left[D-1],right[D],khr[D],D);
            else
                ΦWd     = KhRxTTm(d,left[d-1],right[d+1],khr[d],D);
            end
            # update dth tt-core
            tmp         = ΦWd'*ΦWd + σₙ² *Matrix(I,size(ΦWd,2),size(ΦWd,2));
            tt[d]       = reshape(tmp\(ΦWd'*y),size(tt0[d])) 
            # compute residual
            res[iter,k] = norm(y - ΦWd*tt[d][:])/norm(y)
            # shift norm to next core-to-be-updated
            tt          = shiftTTnorm(tt,d,Dir[k]) 
            # compute new supercore with updated tt-core
            left,right  = getsupercores!(d,left,right,tt[d],khr[d],Dir[k],D)                 
        end
    end
    
    return tt,res
end

function ALS_modelweights(y::Vector,khr::Vector{Matrix},rnks::Vector{Int},maxiter,σₙ²::Float64,dd::Int)
    # This function solves the linear system y = khr*tt with the ALS for the weight w in tensor train format.
    # INPUTS: 
    #   y       observations
    #   khr     matrices that together define a matrix with Khatri-Rao structure, as in eq. (9)
    #   rnks    ranks for the TT
    #   maxiter maximum number of iterations
    #   σₙ²     noise variance (acts as regularization parameter)
    #   dd      first core to be updated       
    
        D           = size(khr,1)   
    ##########################################################################
        # create site-d canonical initial tensor train    
        cores = Vector{Array{Float64,3}}(undef,D);
        for d = 1:dd-1 
            tmp         = qr(rand(rnks[d]*size(khr[d],2), rnks[d+1]));
            cores[d]    = reshape(Matrix(tmp.Q),(rnks[d], size(khr[d],2), rnks[d+1]));
        end
        cores[dd]       = reshape(rand(rnks[dd]*size(khr[dd],2)*rnks[dd+1]),(rnks[dd], size(khr[dd],2), rnks[dd+1]))
        for d = dd+1:D
            tmp         = qr(rand(size(khr[d],2)*rnks[d+1],rnks[d]));
            cores[d]    = reshape(Matrix(tmp.Q)',(rnks[d], size(khr[d],2), rnks[d+1]));
        end
        tt0         = TT(cores,dd);
    ###########################################################################
        # initialize 
        tt          = tt0
        res         = zeros(maxiter,2D-2)
        left,right  = initsupercores(khr,tt0,dd);
        swipe       = [collect(dd:D)...,collect(D-1:-1:2)...,collect(1:dd-1)...];
        Dir         = Int.([ones(1,D-dd)...,-ones(1,D-1)...,ones(1,dd-1)...]);
        covdd       = Matrix{Float64}
        for iter = 1:maxiter
            for k = 1:2D-2
                d           = swipe[k];
                # compute product Φ*W_{\setminus d}
                if d == 1
                    # left[1] is useless, only inputted to not throw error
                    ΦWd     = KhRxTTm(1,left[1],right[2],khr[1],D); 
                elseif d == D
                    # right[D] is useless, only inputted to not throw error
                    ΦWd     = KhRxTTm(D,left[D-1],right[D],khr[D],D);
                else
                    ΦWd     = KhRxTTm(d,left[d-1],right[d+1],khr[d],D);
                end
                # update dth tt-core
                tmp         = ΦWd'*ΦWd + σₙ² *Matrix(I,size(ΦWd,2),size(ΦWd,2));
                tt[d]       = reshape(tmp\(ΦWd'*y),size(tt0[d])) 
                # compute residual
                res[iter,k] = norm(y - ΦWd*tt[d][:])/norm(y)
                # shift norm to next core-to-be-updated
                tt          = shiftTTnorm(tt,d,Dir[k]) 
                # compute new supercore with updated tt-core
                left,right  = getsupercores!(d,left,right,tt[d],khr[d],Dir[k],D)              
            end
        end
        # update dd-th tt-core
        ΦWd         = KhRxTTm(dd,left[dd-1],right[dd+1],khr[dd],D);
        tmp         = ΦWd'*ΦWd + σₙ² *Matrix(I,size(ΦWd,2),size(ΦWd,2));
        tt[dd]      = reshape(tmp\(ΦWd'*y),size(tt0[dd])) 
        covdd       = σₙ²*inv(tmp)
    
        return tt,covdd,res
    end

function ALS_modelweights(X::Matrix,y::Vector,Φ::Vector{Matrix},hyp::Vector,dd::Int,tt::TT)
    # This function solves the linear system y = khr*tt with the ALS for the weight w in tensor train format.
    # INPUTS: 
    #   y       observations
    #   khr     matrices that together define a matrix with Khatri-Rao structure, as in eq. (9)
    #   rnks    ranks for the TT
    #   maxiter maximum number of iterations
    #   σₙ²     noise variance (acts as regularization parameter)
    #   dd      first core to be updated       

    D               = order(tt)   
    ℓ²,σ_f²,σ_n²    = hyp 
    # compute √Λ from hyp
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
    
    left,right      = initsupercores(Φ_,tt,dd);
    swipe           = [collect(dd:D)...,collect(D-1:-1:2)...,collect(1:dd-1)...];
    Dir             = Int.([ones(1,D-dd)...,-ones(1,D-1)...,ones(1,dd-1)...]);
    res             = zeros(2D-2)
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
        tmp         = ΦWd'*ΦWd + σ_n² *Matrix(I,size(ΦWd,2),size(ΦWd,2));
        tt[d]       = reshape(tmp\(ΦWd'*y),size(tt[d])) 
        # compute residual
        res[k] = norm(y - ΦWd*tt[d][:])/norm(y)
        # shift norm to next core-to-be-updated
        tt          = shiftTTnorm(tt,d,Dir[k]) 
        # compute new supercore with updated tt-core
        left,right  = getsupercores!(d,left,right,tt[d],Φ[d],Dir[k],D)              
    end

    # update dd-th tt-core
    ΦWd         = KhRxTTm(dd,left[dd-1],right[dd+1],Φ[dd],D);
    tmp         = ΦWd'*ΦWd + σ_n² *Matrix(I,size(ΦWd,2),size(ΦWd,2));
    tt[dd]      = reshape(tmp\(ΦWd'*y),size(tt[dd])) 

    return tt,ΦWd,res
end

function KhRxTTm(d::Int,leftd::Array{Float64},rightd::Array{Float64},khr::Matrix,D::Int)

    # computes the projected basis functions, Φ*W_{\setminus d} 
    # for ALS to compute model weights of model w = Φ*W_{\setminus d}*w^{(d)}
    
    if d == D 
        pr      = KhatriRao(khr,leftd,1)
    elseif d == 1
        pr      = KhatriRao(rightd,khr,1)
    else
        pr      = KhatriRao(KhatriRao(rightd,khr,1),leftd,1)
    end    
    return pr

end

function KhRxTTm(d::Int,leftd::Array{Float64},rightd::Array{Float64},khr::SparseMatrixCSC{Float64, Int64})

    # computes product of matrix in Khatri-Rao format with matrix in TTm format expressed thru leftd and rightd
    # for ALS to compute model weights of model w = Φ*Wd*Wd
    # for ALS to compute covariance matrix of u for inducing inputs
    N           = size(leftd,1)
    Rd          = size(leftd,2)
    Md          = size(khr,2)

    if d == D 
        pr      = KhatriRao(khr,leftd,1)
    elseif d == 1
        pr      = KhatriRao(rightd,khr,1)
    else
        pr      = KhatriRao(KhatriRao(rightd,khr,1),leftd,1)
    end    
    return pr

end

function getsupercores!(d::Int,left::Vector{Array},right::Vector{Array},ttcore::Array{Float64},khr::Matrix,dir::Int,D::Int)

    if dir == 1 
        if d == 1
            M1              = size(khr,2)
            R2              = size(ttcore,3)
            left[1]         = khr*reshape(ttcore,M1,R2)
        else
            prevsupercore   = left[d-1];
            N               = size(prevsupercore,1)
            Rd              = size(prevsupercore,2)
            Rdd             = size(ttcore,3)   # R_d+1
            Md              = size(khr,2)
            # Khatri-Rao product with next matrix from Khr matrix
            tmp             = KhatriRao(prevsupercore,khr,1)
            Tmp             = reshape(tmp,N,Md,Rd)
            Tmp             = permutedims(Tmp,[1 3 2])
            tmp1            = reshape(Tmp,    N,Rd*Md)
            # contraction with tt-core
            tmp2            = reshape(ttcore,   Rd*Md,Rdd) 
            left[d]         = reshape(tmp1*tmp2,(N,Rdd))
        end
    else 
        if d == D
            MD              = size(khr,2)  
            RD              = size(ttcore,1)
            right[D]        = khr*reshape(ttcore,RD,MD)'
        else
            prevsupercore   = right[d+1];
            N               = size(prevsupercore,1)
            Rdd             = size(prevsupercore,2)  # R_d+1
            Md              = size(khr,2)  
            Rd              = size(ttcore,1)
            # Khatri-Rao product with next matrix from Khr matrix
            tmp1            = KhatriRao(prevsupercore,khr,1)
            # contraction of previous supercore and tt-core
            tmp2            = reshape(ttcore,Rd,Md*Rdd)'
            right[d]        = tmp1*tmp2
        end
    end
    return left,right
end

function getsupercores!(d::Int,left::Vector{Array},right::Vector{Array},ttcore::Array{Float64},khr::SparseMatrixCSC,dir::Int)

    if dir == 1 
        if d == 1
            M1              = size(khr,2)
            R2              = size(ttcore,3)
            left[1]         = khr*reshape(ttcore,M1,R2)
        else
            prevsupercore   = left[d-1];
            N               = size(prevsupercore,1)
            Rd              = size(prevsupercore,2)
            Rdd             = size(ttcore,3)   # R_d+1
            Md              = size(khr,2)
            # Khatri-Rao product with next matrix from Khr matrix
            tmp             = KhatriRao(prevsupercore,khr,1)
            Tmp             = reshape(tmp,N,Md,Rd)
            Tmp             = permutedims(Tmp,[1 3 2])
            tmp1            = reshape(Tmp,    N,Rd*Md)
            # contraction with tt-core
            tmp2            = reshape(ttcore,   Rd*Md,Rdd) 
            left[d]         = reshape(tmp1*tmp2,(N,Rdd))
        end
    else 
        if d == D
            MD              = size(khr,2)  
            RD              = size(ttcore,1)
            right[D]        = khr*reshape(ttcore,RD,MD)'
        else
            prevsupercore   = right[d+1];
            N               = size(prevsupercore,1)
            Rdd             = size(prevsupercore,2)  # R_d+1
            Md              = size(khr,2)  
            Rd              = size(ttcore,1)
            # Khatri-Rao product with next matrix from Khr matrix
            tmp1            = KhatriRao(prevsupercore,khr,1)
            # contraction of previous supercore and tt-core
            tmp2            = reshape(ttcore,Rd,Md*Rdd)'
            right[d]        = tmp1*tmp2
        end
    end
    return left,right
end

function initsupercores(khr::Vector{Matrix},tt0::TT)
    # initializes left and right supercores for a the first update in the ALS (last core)
    # works yay :) 
    D           = size(khr,1)
    M1          = size(khr[1],2)
    R2          = size(tt0[1],3)
    MD          = size(khr[D],2)
    RD          = size(tt0[D],1)
    left        = Vector{Array}(undef,D)
    right       = Vector{Array}(undef,D)
    left[1]     = khr[1]*reshape(tt0[1],M1,R2)
    for d = 2:D-1
        left,right = getsupercores!(d,left,right,tt0[d],khr[d],1,D)
    end
    right[D]   = khr[D]*reshape(tt0[D],RD,MD)'
    return left,right
end

function initsupercores(khr::Vector{Matrix},tt0::TT,dd::Int)
    # initializes left and right supercores for a the first update in the ALS (dd-th core)
    # works yay :) 
    D           = size(khr,1)
    M1          = size(khr[1],2)
    R2          = size(tt0[1],3)
    MD          = size(khr[D],2)
    RD          = size(tt0[D],1)
    left        = Vector{Array}(undef,D)
    right       = Vector{Array}(undef,D)
    left[1]     = khr[1]*reshape(tt0[1],M1,R2)
    for d = 2:dd-1
        left,right = getsupercores!(d,left,right,tt0[d],khr[d],1,D)
    end
    right[D]   = khr[D]*reshape(tt0[D],RD,MD)'
    for d = D-1:-1:dd+1
        left,right = getsupercores!(d,left,right,tt0[d],khr[d],-1,D)
    end

    return left,right
end

end