module functionsTT

using LinearAlgebra
import Base: transpose 

export TT, TTv,TTm, size, norm, order, rank, shiftTTnorm, ttv2vec, ttm2mat, TTv_SVD, TTm_SVD

mutable struct TT{N}
    cores::Vector{Array{Float64,N}}
    normcore::Int64
    function TT(cores::Vector{Array{Float64,N}},normcore::Int64) where N
        new{ndims(cores[1])}(cores,normcore) 
    end
end
TT(cores) = TT(cores,0);

# aliases for MPS and MPO
const TTv = TT{3};
const TTm = TT{4};

# indexing to get and set a core in an TT
Base.IndexStyle(::Type{<:TT}) = IndexLinear() 
Base.getindex(tt::TT, i::Int) = tt.cores[i] # tt[1] gives the first core
Base.getindex(tt::TT, range::UnitRange{Int64}) = [tt.cores[i] for i in range]
Base.setindex!(tt::TT,v,i::Int) = setindex!(tt.cores, v, i) # tt[1] = rand(1,5,3) sets the first core

function Base.size(tt::TTv)
    [size(core)[2] for core in tt.cores];
end

function Base.size(ttm::TTm)
    sizes = Int.(zeros(2,order(ttm)))
    for d = 1:order(ttm)
        sizes[1,d] = size(ttm.cores[d],2)
        sizes[2,d] = size(ttm.cores[d],3)
    end
    return sizes
end

function order(tt::TT)
    collect(size(tt.cores))[1]
end

function LinearAlgebra.rank(tt::TT)
    sizes = [size(core) for core in tt.cores];
    [[sizes[i][1] for i in 1:length(sizes)]..., 1]    
end

function shiftTTnorm(tt::TTv,d::Int64,dir::Int64)

    if dir == 1
        sztt    = size(tt[d])
        Gl      = reshape(tt[d],sztt[1]*sztt[2],sztt[3])
        F       = qr(Gl);
        R       = Matrix(F.R); Q = Matrix(F.Q);
        sztt    = size(tt[d+1])
        tt[d+1] = reshape(R*reshape(tt[d+1],sztt[1],sztt[2]*sztt[3]),sztt)
    elseif dir == -1
        sztt    = size(tt[d])
        Gr      = reshape(tt[d],sztt[1],sztt[2]*sztt[3])
        F       = qr(Gr');
        R       = Matrix(F.R)'; Qt = Matrix(F.Q);
        Q       = Qt';
        sztt    = size(tt[d-1])
        tt[d-1] = reshape(reshape(tt[d-1],sztt[1]*sztt[2],sztt[3])*R,sztt)
    end

    tt[d]       = reshape(Q, size(tt[d]));
    tt.normcore = tt.normcore + dir; 
    return tt
end

function transpose(ttm::TTm)
    N     = order(ttm);
    cores = Vector{Array{Float64,4}}(undef,N);
    for i = 1: order(ttm)
        cores[i] = permutedims(ttm[i],[1,3,2,4]);
    end
    return TT(cores)
end

# reconstruction of vector represented by an TTv
function ttv2vec(tt::TTv)
    tensor = reshape(tt[1],size(tt[1],1)*size(tt[1],2),size(tt[1],3))
    for i = 2:order(tt)
        tensor = tensor*reshape(tt[i],size(tt[i],1),size(tt[i],2)*size(tt[i],3))
        tensor = reshape(tensor, (Int(length(tensor)/size(tt[i],3)), size(tt[i],3)));
    end
    vector = tensor
    return vector
end

# reconstruction of matrix represented by an TTm
function ttm2mat(ttm::TTm)
    tensor = reshape(ttm[1],size(ttm[1],1)*size(ttm[1],2)*size(ttm[1],3),size(ttm[1],4))
    sizes  = size(ttm);
    D      = order(ttm)

    for i = 2:D
        tensor = tensor*reshape(ttm[i],size(ttm[i],1),size(ttm[i],2)*size(ttm[i],3)*size(ttm[i],4))
        tensor = reshape(tensor, (Int(length(tensor)/size(ttm[i],4)), size(ttm[i],4)));
    end
    tensor = reshape(tensor,(sizes[1,:]...,sizes[2,:]...));
    tensor = permutedims(tensor,[collect(1:2:2D-1)..., collect(2:2:2D)...]);
    matrix = reshape(tensor,(prod(sizes[1,:]),prod(sizes[2,:])));
    return matrix
end

function TTm_SVD(mat::Matrix,middlesizes::Matrix,acc::Float64)
    sizes   = Tuple(reshape(middlesizes',(length(middlesizes),1)));
    tensor  = reshape(mat,sizes);
    permind = [ (i-1)*size(middlesizes,2)+j for j in 1:size(middlesizes,2) for i in 1:2 ];
    tensor  = permutedims(tensor,permind);
    resind  = Tuple([prod(col) for col in eachcol(middlesizes)]);
    tensor  = reshape(tensor,resind)
    tt,err  = TT_SVD(tensor,acc);
    rnks    = rank(tt);
    return TT( [reshape(tt[i],(rnks[i], middlesizes[:,i]..., rnks[i+1])) for i = 1:order(tt)] ),err
end

function TTv_SVD(vec::Vector,middlesizes::Vector,acc::Float64)
    tensor  = reshape(vec,Tuple(middlesizes));
    return TT_SVD(tensor,acc);
end

function TT_SVD(tensor::Array{Float64},ϵ::Float64)
    ########################################################################    
    #   Computes the cores of a TT for the given tensor and accuracy (acc)     
    #   Resources:
    #   V. Oseledets: Tensor-Train Decomposition, 2011, p.2301: Algorithm 1
    #   April 2021, Clara Menzen
    ########################################################################
        D           = ndims(tensor);
        cores       = Vector{Array{Float64,3}}(undef,D);
        frobnorm    = norm(tensor); 
    
        δ = ϵ / sqrt(D-1) * frobnorm;
        err2 = 0;
        rprev = 1;
        sizes = size(tensor);
        C = reshape( tensor, (sizes[1], Int(length(tensor) / sizes[1]) ));
        for k = 1 : D-1
            # truncated svd 
            F   = svd!(C); 
            rcurr = length(F.S);
    
            sv2 = cumsum(reverse(F.S).^2);
            tr  = Int(findfirst(sv2 .> δ^2))-1;
            if tr > 0
                rcurr = length(F.S) - tr;
                err2 += sv2[tr];
            end
            
            # new core
            cores[k] = reshape(F.U[:,1:rcurr],(rprev,sizes[k],rcurr));
            rprev    = rcurr;
            C        = Diagonal(F.S[1:rcurr])*F.Vt[1:rcurr,:];
            C        = reshape(C,(rcurr*sizes[k+1], Int(length(C) / (rcurr*sizes[k+1])) ) );
        end
        cores[D] = reshape(C,(rprev,sizes[D],1));
        return TT(cores,D), sqrt(err2)/frobnorm
    end

end