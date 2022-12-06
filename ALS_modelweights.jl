function ALS_modelweights(y::Vector,khr::Vector{Any},rnks::Vector{Int},maxiter,λ::Float64)
# This function solves the linear system y = khr*w with the ALS for the weight w in tensor train format.
# INPUTS: 
#   y       observations
#   khr     matrices that together define a matrix with Khatri-Rao structure, as in eq. (9)
#   rnks    ranks for the TT
#   maxiter maximum number of iterations
#   λ       regularization parameter

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
    tt0         = MPT(cores,D);
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
                # left[d] is useless, only inputted to not throw error
                ΦWd     = getprojectedKhR(d,left[1],right[2],khr[1]); 
            elseif d == D
                # right[d] is useless, only inputted to not throw error
                ΦWd     = getprojectedKhR(D,left[D-1],right[D],khr[D]);
            else
                ΦWd     = getprojectedKhR(d,left[d-1],right[d+1],khr[d]);
            end
            # update dth tt-core
            tmp         = ΦWd'*ΦWd + λ*Matrix(I,size(ΦWd,2),size(ΦWd,2));
            #tt[d]       = reshape(pinv(ΦWd)*y,size(tt0[d])) # how to use pinv with regularization?
            tt[d]       = reshape(tmp\(ΦWd'*y),size(tt0[d])) 
            # compute residual
            res[iter,k] = norm(y - ΦWd*tt[d][:])/norm(y)
            # shift norm to next core-to-be-updated
            tt          = shiftMPTnorm(tt,d,Dir[k]) 
            # compute new supercore with updated tt-core
            left,right  = getsupercores!(d,left,right,tt[d],khr[d],Dir[k])                 
        end
    end
    
    return tt,res
end

function getprojectedKhR(d::Int,leftd::Array{Float64},rightd::Array{Float64},khr::Matrix)

    # computes the projected basis functions, Φ*W_{\setminus d} 
    # for ALS to compute model weights of model w = Φ*W_{\setminus d}*w^{(d)}
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

function getprojectedKhR(d::Int,leftd::Array{Float64},rightd::Array{Float64},khr::SparseMatrixCSC)

    # computes the projected basis functions, Φ*W_{\setminus d} 
    # for ALS to compute model weights of model w = Φ*W_{\setminus d}*w^{(d)}
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

function getsupercores!(d::Int,left::Vector{Array},right::Vector{Array},ttcore::Array{Float64},khr::Matrix,dir::Int)

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

function initsupercores(khr::Vector{Any},tt0::MPT{3})
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
        left,right = getsupercores!(d,left,right,tt0[d],khr[d],1)
    end
    right[D]   = khr[D]*reshape(tt0[D],RD,MD)'
    return left,right
end

function initsupercores(khr::Vector{SparseMatrixCSC},tt0::MPT{3})
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
        left,right = getsupercores!(d,left,right,tt0[d],khr[d],1)
    end
    right[D]   = khr[D]*reshape(tt0[D],RD,MD)'
    return left,right
end

function initsupercores(khr::Vector{Any},tt0::MPT{3},bool::Bool)
    D           = size(khr,1)
    left        = Vector{Array}(undef,D)
    right       = Vector{Array}(undef,D)
    MD          = size(khr[D],2)
    RD          = size(tt0[D],1)
    if bool == true # start with site-1
        right[D]     = khr[D]*reshape(tt0[D],RD,MD)'
        for d = D-1:-1:2
            left,right = getsupercores!(d,left,right,tt0[d],khr[d],-1)
        end
    end
    return left,right
end

function getWd(tt::MPT{3},d::Int)
    # computes the matrix W_{∖setminus d} in TT matrix format
    D           = order(tt)
    middlesizes = size(tt,true)
    M           = middlesizes[d]
    newms       = zeros(2,D)
    newms[1,:]  = middlesizes
    newms[2,:]  = ones(D)
    Wd          = mps2mpo(tt,Int.(newms))
    Wd[d]       = reshape(Matrix(I,(M,M)),(1,M,M,1))
    if d>1
        Wd[d-1] = permutedims(Wd[d-1],(1,2,4,3))
    end
    if d<D
        Wd[d+1] = permutedims(Wd[d+1],(3,2,1,4))
    end
    return Wd
end