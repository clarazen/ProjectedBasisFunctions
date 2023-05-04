module functionsALSmodelweights

using LinearAlgebra
using SparseArrays
using ..functionsTT
using ..functions_KhatriRao_Kronecker

export ALS_modelweights,initsupercores,KhRxTTm,penalmat,initTT

function ALS_modelweights(y,khr,maxiter,λ,σ_y²,tt)
    # This function solves the linear system y = khr*tt with the ALS for the weight w in tensor train format.
    # an initial TT is inputted

    D                   = order(tt)   
    dd                  = tt.normcore

    # initialize
    left,right          = initsupercores(khr,tt,dd);
    swipe               = [collect(dd:D)...,collect(D-1:-1:2)...,collect(1:dd-1)...];
    Dir                 = Int.([ones(1,D-dd)...,-ones(1,D-1)...,ones(1,dd-1)...]);
    covdd               = Matrix{Float64}
    
    # compute tt
    res                 = zeros(maxiter,2D-2)
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
            if λ == 0.0
                tt[d]       = reshape(pinv(ΦWd'*ΦWd)*(ΦWd'*y),size(tt[d])) 
            else
                tt[d]       = reshape((ΦWd'*ΦWd + λ*I)\(ΦWd'*y),size(tt[d])) 
                # compute penalty matrix
                #P       = diff(I(size(khr[1],2)),dims=1);
                #PP      = P'*P;
                #Wpen    = penalmat(tt,d,D,P,PP)
                # Weighted sum of the difference penalty matrices
                #WWW = λ*Wpen[1];
                #for i =2:D
                #    WWW = WWW + λ*Wpen[i];
                #end

                #tt[d]      = reshape((ΦWd'*ΦWd + WWW)\(ΦWd'*y),size(tt[d]))
            end
            
            # compute residual
            res[iter,k] = norm(y - ΦWd*tt[d][:])/norm(y)
            # shift norm to next core-to-be-updated
            tt          = shiftTTnorm(tt,d,Dir[k]) 
            # compute new supercore with updated tt-core
            left,right  = getsupercores!(d,left,right,tt[d],khr[d],Dir[k],D)              
        end
    end
    
    # update dd-th tt-core
    ΦWd                 = KhRxTTm(dd,left[dd-1],right[dd+1],khr[dd],D);
    if λ == 0
        tmp             = pinv(ΦWd'*ΦWd)
        tt[dd]          = reshape(tmp*(ΦWd'*y),size(tt[dd])) 
        covdd           = σ_y²*tmp
    else
        P       = diff(I(size(khr[1],2)),dims=1);
        PP      = P'*P;
        Wpen    = penalmat(tt,dd,D,P,PP)
        # Weighted sum of the difference penalty matrices
        WWW = λ*Wpen[1];
        for i =2:D
            WWW = WWW + λ*Wpen[i];
        end

        #tt[dd]      = reshape((ΦWd'*ΦWd + WWW)\(ΦWd'*y),size(tt[dd]))
        tt[dd]          = reshape((ΦWd'*ΦWd + λ*I)\(ΦWd'*y),size(tt[dd])) 
        #covdd           = σ_y²*inv(ΦWd'*ΦWd + λ*I)
    end

    return tt,covdd,res,ΦWd
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


function initsupercores(khr,tt0::TT,dd::Int)
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

function initTT(rnks,Md,dd,D)
    # create site-d canonical initial tensor train    
    cores = Vector{Array{Float64,3}}(undef,D);
    for d = 1:dd-1 
        tmp         = qr(rand(rnks[d]*Md, rnks[d+1]));
        cores[d]    = reshape(Matrix(tmp.Q),(rnks[d], Md, rnks[d+1]));
    end
    cores[dd]       = reshape(rand(rnks[dd]*Md*rnks[dd+1]),(rnks[dd], Md, rnks[dd+1]))
    for d = dd+1:D
        tmp         = qr(rand(Md*rnks[d+1],rnks[d]));
        cores[d]    = reshape(Matrix(tmp.Q)',(rnks[d], Md, rnks[d+1]));
    end
    return TT(cores,dd);
end


function penalmat(tt::TT,sweepindex::Int,D::Int64,P::Matrix{Int64},PP::Matrix{Int64})

    DD              = Vector{Matrix}(undef,D)
    W               = Vector{Matrix}(undef,D)
    DWD             = Vector{Matrix}(undef,D)
    eyez            = Vector{Any}(undef,D+1)
    eyep            = Vector{Any}(undef,D)
    for d = 1:D
        Csize       = size(tt[d]);
        
        Dm          = reshape(permutedims(tt[d], [2, 1, 3]), (Csize[2], Csize[1]*Csize[3]));
        mDDm        = reshape(Dm'*Dm, (Csize[1], Csize[3], Csize[1], Csize[3]));                      #O(I*r^4)
        DD[d]       = reshape(permutedims(mDDm,[1, 3, 2, 4]), (Csize[1]*Csize[1], Csize[3]*Csize[3]));
        PD          = P*Dm;                                                                          #O(I^2*r^2)
        DPPD        = reshape(PD'*PD, (Csize[1], Csize[3], Csize[1], Csize[3]));                      #O(I^2*r^4)
        DWD[d]      = reshape(permutedims(DPPD,[1, 3, 2, 4]), (Csize[1]*Csize[1], Csize[3]*Csize[3]));   
        eyez[d]     = I(Csize[1])[:]
        eyep[d]     = I(Csize[2])[:]
    end
    eyez[D+1]       = 1;
    
    for d = 1:D   #O(d^2*r^4)
        Dsize       = size(tt[sweepindex])
            if sweepindex == d
                D1  = eyez[sweepindex];
                D2  = PP[:];
                D3  = eyez[sweepindex+1];
            elseif sweepindex < d
                D1  = eyez[sweepindex];  
                D2  = eyep[sweepindex];   
                D3  = DWD[d]*eyez[d+1];           
                for it=(d-1):-1:(sweepindex+1)               
                    D3 = DD[it]*D3;                     #O(d*r^4)
                end                               
            elseif sweepindex > d   
                D1  = eyez[d]'*DWD[d];
   
                for it=(d+1):(sweepindex-1)               
                    D1 = D1*DD[it];                     #O(d*r^4)
                end
                D1  = D1';
                D2  = eyep[sweepindex];   
                D3  = eyez[sweepindex+1]; 
            end
      
        WW          = kron(D3, kron(D2, D1)); #O(I^4*r^4)
        Wtemp       = permutedims(reshape(WW, (Dsize[1], Dsize[1], Dsize[2], Dsize[2], Dsize[3], Dsize[3])), [1, 3, 5, 2, 4, 6]);
        W[d]        = reshape(Wtemp, (prod(Dsize), prod(Dsize))); 
    end

    return W

end

end