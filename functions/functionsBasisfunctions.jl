module functionsBasisfunctions

using LinearAlgebra
using SparseArrays
using ..functions_KhatriRao_Kronecker
using SpecialFunctions

export basisfunctionsSE!,basisfunctionsMatern!,bsplines

function basisfunctionsSE!(M::Vector{Float64},X::Matrix{Float64},ℓ::Float64,σ_f::Float64,L::Vector{Float64},Φ_)
    # computes Φ_, such that Φ_*Φ_' approx K as Khatri Rao multipliers
        D = size(X,2)
        for d = 1:D
            w        = collect(1:M[d])';
            sqrtΛ    = sqrt.(σ_f^(1/D)*sqrt(2π*ℓ) .* exp.(- ℓ/2 .* ((π.*w')./(2L[d])).^2 ))
            Φ_[d]    = (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w).*sqrtΛ';
        end
    
        return Φ_
end

function basisfunctionsSE!(M::Vector{Float64},X::Matrix{Float64},ℓ::Float64,σ_f::Float64,L::Vector{Float64},Φ,Λ)
    # computes Φ (as Khatri Rao multipliers) and 𝝠 such that Φ*Λ*Φ' approx K_SE 
        D = size(X,2)
        for d = 1:D
            w           = collect(1:M[d])';
            Λ[d]        = σ_f^(1/D)*sqrt(2π*ℓ) .* exp.(- ℓ/2 .* ((π.*w')./(2L[d])).^2 ) 
            Φ[d]        = (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w);
        end
    
        return Φ,Λ
end 

function basisfunctionsMatern!(M::Vector{Float64},X::Matrix{Float64},ℓ²::Float64,σ_f²::Float64,L::Vector{Float64},Φ,Λ,ν)
    # computes Φ (as Khatri Rao multipliers) and 𝝠 such that Φ*Λ*Φ' approx K_matern of degree deg
        D = size(X,2)
        for d = 1:D
            w           = collect(1:M[d])';
            Λ[d]        = σ_f²^(1/D)*2*π^(1/2)*(2ν)^ν/ℓ²^ν * (gamma(ν+1/2,0)/gamma(ν,0)) .* ((2ν/ℓ² .+ 4π^2*((π.*w')./(2L[d])).^2).^(-ν+1/2))
            Φ[d]        = (1/sqrt(L[d])) .*sinpi.(  ((X[:,d].+L[d])./2L[d]).*w);
        end
    
        return Φ,Λ
end 

Γfunc(x) = prod(collect(x-1:1))

function basisfunctionsSE!(budget::Int,X::Matrix{Float64},ℓ::Float64,σ_f::Float64,L::Vector)
        # computes Φ and 𝝠 such that Φ*sqrtΛ * sqrtΛ*Φ' approx K in matrix format
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
       
        return ΦR,ΛR,ind
end

function bsplines(X,ρ,knotint)
    N,D         = size(X)
    A           = basismat(ρ)
    knotdist    = 1/knotint
    ind         = Int.(floor.(X ./ knotdist) .+ 1)
    ind[ ind .> knotint ].= knotint;
    ind[ ind .< 1 ].= 1;

    inputs      = (X ./ knotdist) .-ind.+1;

    Φ           = Vector{Matrix}(undef,D)
    for d=1:D
        bn = inputs[:,d].^reshape(collect(ρ:-1:0),1,ρ+1)*A;   # Construct the nonzero elements of the b-spline basis vectors using the matrix form.
        
        Φ[d]    = zeros(N,ρ+knotint);
        for n = 1:N
           Φ[d][n,ind[n,d]:ind[n,d]+ρ] = bn[n,:]; #Store them in the correct location within the basis vector. 
        end
    end

    return Φ

end

function basismat(ρ::Int)
# based on implemetation by Karagoz
# which is based on 'General matrix representations for B-splines', - Kaihuai Qin
    A        = Vector{Any}(undef,ρ+1)
    A[1]     = 1;

    for k = 2:ρ+1

        D1   = diagm(collect(1:k-1))
        D2   = Bidiagonal(zeros(size(collect(k-2:-1:0),1)+1),collect(k-2:-1:0),:U)
        D    = [D1 zeros(k-1,1)] + D2[1:k-1,:]
        A[k] = 1/(k-1) *( [A[k-1] ; zeros(1,k-1)] * D + [zeros(1,k-1) ; A[k-1]] * diff(Matrix(I,k,k),dims=1))
    end

    return reverse(A[ρ+1],dims=1)
end



end

