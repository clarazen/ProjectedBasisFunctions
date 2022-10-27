function khrxttm(khr::Vector{Matrix},ttm::MPT{4},d::Int,lr::Vector{Array{Float64}})
    # computes the produt of two matrices, where the first one (khr) has a 
    # row-wise Khatri-Rao structure and the second (ttm) is a TTm for finding solution of linear system.
    # d is the TT-core that is updated
    # all other core-by-core multiplications are precomputed

    D  = order(ttm)
    N  = size(kr[1],1)
    
    Tmp     = nmodeproduct(kr[1],ttm[1][1,:,:,:],2)
    tmp     = KhatriRao(unfold(Tmp,[2],"right"),Matrix(kr[2]'),2) 
    newdims = (size(ttm[2],3),size(ttm[1],2),size(ttm[1],4),N)
    Tmp     = reshape(tmp,newdims)
    Tmp     = contractmodes(Tmp,ttm[2],[1 3; 3 1]) #this is slow

    for d = 2:D-1     
        tmp     = KhatriRao(unfold(Tmp,[d],"right"),Matrix(kr[d+1]'),2)
        newdims = (size(ttm[d+1],3),newdims[2:d]...,size(ttm[d],2),size(ttm[d],4),N)
        Tmp     = reshape(tmp,newdims)
        Tmp     = contractmodes(Tmp,ttm[d+1],[1 3; d+2 1])  #this is slow
    end
    return unfold(Tmp,[D],"right")
end