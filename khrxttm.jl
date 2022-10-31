function khrxttm(d::Int,leftd::Array{Float64},rightd::Array{Float64})

    # compute product
    Md      = size(leftd,1)
    N       = size(leftd,d+2)
    Rd      = size(leftd,d+1)
    Rdd     = size(rightd,ndims(rightd)-1)
    Tmp     = permutedims(leftd,[collect(2:d+1)...,1,d+2])
    left    = reshape(Tmp,(length(Tmp)/(Md*M), Md*M))
    Tmp     = permutedims(rightd,[collect(2:d+1)...,1,d+2])
    right   = reshape(Tmp,(length(Tmp)/(Md*M), Md*M))

    tmp     = KhatriRao(left,right,2)
    Tmp     = reshape(tmp,(Rdd,Rd,Md,N))
    pr      = permutedims(Tmp,[2,3,1,4])

    return pr

end

function getsupercore(d::Int,left::Vector{Array},right::Vector{Array},newcore::Array{Float64},khr::Matrix,dir::Int)

    if dir == 1 
        prevsupercore   = left[d-1];
        # new super core is previous left super core contracted with updated core
        Md              = size(prevsupercore,1)
        O               = size(prevsupercore,2:d)   # O_1 ... O_d-1
        Rd              = size(prevsupercore,d+1)
        N               = size(prevsupercore,d+2)
        Tmp             = permute(prevsupercore,[collect(2:d)...,d+2,1,d+1])
        tmp1            = reshape(Tmp, ((prod(O)*N)/(Md*Rd),Md*Rd))

        Od              = size(newcore,2)
        Rdd             = size(newcore,4)
        Tmp             = permutedims(newcore,[3,1,2,4])
        tmp2            = reshape(Tmp,(Md*Rd,(Od*Rdd)/(Md*Rd)))
        Tmp             = reshape(tmp1*tmp2,[O...,N,Od,Rdd])
        # Khatri-Rao product with next matrix from Khr matrix
        Tmp             = permutedims(Tmp,(collect(1:d-1)...,d+1,d+2,d))
        tmp             = reshape(Tmp,(prod(O)*Od*Rdd,N))
        tmp             = KhatriRao(tmp,khr,2)
        left[d]         = reshape(tmp,(size(khr,2),O...,Od,Rdd,N))
        return left
    else 
        prevsupercore   = right[d-1];
        # new super core is previous right super core contracted with updated core
        Md              = size(prevsupercore,1)     # M_d
        O               = size(prevsupercore,2:d-1) # O_D ... O_d+2
        Rdd             = size(prevsupercore,d)     # R_d+1
        Odd             = size(prevsupercore,d+1)   # O_d+1
        N               = size(prevsupercore,d+2)   # N
        Tmp             = permute(prevsupercore,[collect(2:d-1)...,d,d+1,1,d+2])
        tmp1            = reshape(Tmp, ((prod(O)*Odd*N)/(Md*Rd),Md*Rd))

        Od              = size(newcore,2)
        Rdd             = size(newcore,4)
        Tmp             = permutedims(newcore,[3,4,1,2])
        tmp2            = reshape(Tmp,(Md*Rdd,(Od*Rd)/(Md*Rdd)))
        Tmp             = reshape(tmp1*tmp2,[O...,N,Od,Rd])
        # Khatri-Rao product with next matrix from Khr matrix
        Tmp             = permutedims(Tmp,(collect(1:d-2)...,d,d+1,d-1))
        tmp             = reshape(Tmp,(prod(O)*Odd*Rd,N))
        tmp             = KhatriRao(tmp,khr,2)
        # reshape one more time i think
        right[d]        = reshape(tmp,(size(khr,2),O...,Od,Rd,N))
        return right
    end

end

function initsupercores(left::Vector{Array},right::Vector{Array},ttm0::MPT{4},khr::Vector{Matrix})
    # initializes leftd and right d for a the first update of the last core
    left[1] = ones();
   for d = 1:D
        left[d+1] = getsupercore(d,left,right,ttm0[d],khr[d],1)
   end
   right[D] = ones(); 
end