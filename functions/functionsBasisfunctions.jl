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

function colectofbasisfunc(M::Vector{Float64},X::Matrix{Float64},ℓ::Float64,σ_f::Float64,L::Vector{Float64},eig)
    # computes Φ and 𝝠 such that Φ*Λ*Φ' approx K
        D = size(X,2)
        Φ = Vector{Matrix}(undef,D);
        Λ = Vector{Vector}(undef,D);
        for d = 1:D
            w     = collect(1:M[d])';
            Λ[d]  = σ_f^(1/D)*sqrt(2π*ℓ) .* exp.(- ℓ/2 .* ((π.*w')./(2L[d])).^2 )
            Φ[d] = (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w);
        end
    
        return Φ,Λ
end 

function colectofbasisfunc(budget::Int,X::Matrix{Float64},ℓ::Float64,σ_f::Float64,L::Vector)
        # computes leading eigenfunctions
        N       = size(X,1)
        D       = size(X,2)
        M       = Int(ceil(budget^(1/D)));

        Λ       = 1;
        for d = D:-1:1
            w           = collect(1:M)';
            Λ           = kron(Λ,spdiagm(σ_f^(1/D)*sqrt(2π*ℓ) .* exp.(- ℓ/2 .* ((π.*w')./(2L[d])).^2 )))
        end
        p       = sortperm(diag(Λ),rev=true)[1:budget] # budget values of sorting permutation
        ΛR      = Λ[p,p] # budget eigenvalues
        allind  = Tuple(vcat([1:M for d in 1:D]))
        ind     = Float64.(Matrix(reshape(reinterpret(Int,CartesianIndices(allind)[p]),D,budget)))

        ΦR      = ones(N,budget)
        for d = 1:D
            ΦR      = ΦR .* (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*ind[d,:]')
        end
       
        return ΦR,ΛR
end

end

