module functionsBasic

using LinearAlgebra, SparseArrays

export gensynthdata, gengriddata, covSE, fullGP

function gensynthdata(N::Int64,D::Int64,hyp::Vector)
    σ_n   = sqrt(hyp[3]);
    X     = zeros(N,D);
    jitter = sqrt(eps(1.))
    for d = 1:D
        for n = 1:N
            X[n,d] = rand(1)[1].* 2 .-1
        end
    end
    K      = covSE(X,X,hyp);
    f      = Matrix(cholesky(K+jitter*Matrix(I,size(K))))*randn(N);
    y      = f + σ_n*randn(size(f,1));
    return X, y, f, K 
end

function covSE(Xp::Matrix{Float64},Xq::Matrix{Float64},hyp::Vector{Float64})
    ℓ     = hyp[1];
    σ_f   = hyp[2];
    D     = size(Xp,2)

    K = zeros(size(Xp,1),size(Xq,1))
    for i = 1:size(Xp,1)
        for j = 1:size(Xq,1)
            exparg = norm(Xp[i,:]-Xq[j,:])^2/2ℓ
            K[i,j] = σ_f * exp(-exparg)
        end
    end
    return K
end

function fullGP(K::Matrix,X::Matrix,Xstar::Matrix,y::Vector,hyp::Vector)

    σ_n   = hyp[3];
    N     = size(X,1)
    
    L     = cholesky(K+(σ_n+sqrt(eps(1.0)))*Matrix(I,N,N)).L;
    Ks    = covSE(Xstar,X,hyp);
    Kss   = covSE(Xstar,Xstar,hyp);
    α     = L'\(L\y);
    mstar = Ks*α;
    v     = L\Ks';
    Pstar = Kss - v'*v;
    vstar = diag(Pstar);
    
    return mstar, vstar
end
    
function gengriddata(Md::Int,D::Int,min::Vector,max::Vector,m::Bool)
    coord = Vector{Vector}(undef,D)
    X     = spzeros(Md^D,D)
    for d = 1:D
        coord[d] = range(min[d],max[d],length=Md) # coordinate in dth dimension
    end
    Coord = Tuple(coord)
    if m == true
        i=1;
        for d = D:-1:1
            X[:,i] = getindex.(Iterators.product(Coord...), d)[:]
            i = i+1
        end
        return X,coord
    else
        return coord
    end
end

end