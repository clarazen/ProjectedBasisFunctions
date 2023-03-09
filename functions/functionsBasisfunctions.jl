module functionsBasisfunctions

using LinearAlgebra
using SparseArrays
using ..functions_KhatriRao_Kronecker

export colectofbasisfunc

function colectofbasisfunc(M::Vector{Float64},X::Matrix{Float64},ℓ::Float64,σ_f::Float64,L::Vector{Float64})
    # computes Φ_, such that Φ_*Φ_' approx K
        D = size(X,2)
        Φ_ = Vector{Matrix}(undef,D);
        for d = 1:D
            w     = collect(1:M[d])';
            Λ     = σ_f^(1/D)*sqrt(2π*ℓ) .* exp.(- ℓ/2 .* ((π.*w')./(2L[d])).^2 )
            Φ_[d] = (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w).*sqrt.(Λ)';
        end
    
        return Φ_
end

    

function colectofbasisfunc(budget::Int,X::Matrix{Float64},ℓ::Float64,σ_f::Float64,L::Vector)
        # computes leading eigenfunctions
        N = size(X,1)
        D = size(X,2)
        M = Int(ceil(budget^(1/D)));

        Λ = 1;
        for d = D:-1:1
            w    = collect(1:M)';
            Λ    = kron(Λ,spdiagm(σ_f^(1/D)*sqrt(2π*ℓ) .* exp.(- ℓ/2 .* ((π.*w')./(2L[d])).^2 )))
        end
        p   = sortperm(diag(Λ),rev=true)[1:budget] # budget values of sorting permutation
        ΛR  = Λ[p,p] # budget eigenvalues
        ind = Float64.(Matrix(reshape(reinterpret(Int,CartesianIndices((1:M,1:M,1:M))[p]),D,budget)))

        ΦR   = ones(N,budget)
        for d = 1:D
            ΦR = ΦR .* (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*ind[d,:]')
        end
       
        return ΦR,ΛR
end

#=
function colectofbasisfunc(M::Vector,X::Matrix,ℓ::Float64,σ_f::Float64,L::Vector)
 # computes Φ,S,Λ such that Φ*S*S*Φ' approx K and Λ contains inserse eigenvalues
    D = size(X,2)
    
    Φ     = Vector{Matrix}(undef,D);
    invΛ  = Vector{Matrix}(undef,D);
    S     = Vector{Matrix}(undef,D);
    for d = 1:D
        w       = collect(1:M[d])';
        tmp     = σ_f^(1/D)*sqrt(2π*ℓ) .* exp.(- ℓ/2 .* ((π.*w')./(2L[d])).^2 )
        invΛ[d] = Diagonal(1 ./ tmp)
        S[d]    = Diagonal(sqrt.(tmp));
        Φ[d]    = (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w);
    end

    return Φ,kr2ttm(Λ),kr2ttm(S)
end

function colectofbasisfunc(Φ::Vector{Matrix},S::MPT{4},trunc::Float64)
    
    Φmat = khr2mat(Φ)
    ev   = diag(mpo2mat(S)).^2
    perm = sortperm(ev,rev=true)
    ev   = sort(ev,rev=true)
    indx = findfirst(ev.<trunc)

    Φmat = Φmat[:,perm[1:indx]]
    Λ    = diagm(1 ./ ev[1:indx])
    
    return Φmat,Λ
end
=#

end

