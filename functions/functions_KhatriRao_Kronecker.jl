module functions_KhatriRao_Kronecker

using LinearAlgebra

export khr2mat,KhatriRao

function khr2mat(Φ::Vector{Matrix})
    # computes the row-wise Khatri-Rao product for given set of matrices
    Φ_mat = ones(size(Φ[1],1),1)
    for d = size(Φ,1):-1:1
        Φ_mat = KhatriRao(Φ_mat,Φ[d],1)
    end    
    return Φ_mat
end

function KhatriRao(A::Matrix{Float64},B::Matrix{Float64},dims::Int64)
    if dims == 1 # row-wise
        C = zeros(size(A,1),size(A,2)*size(B,2));
        @inbounds @simd for i = 1:size(A,1)
            @views kron!(C[i,:],A[i,:],B[i,:])
        end
    elseif dims == 2 # column-wise
        C = zeros(size(A,1)*size(B,1),size(A,2));
        @inbounds @simd for i = 1:size(A,2)
            @views kron!(C[:,i],A[:,i],B[:,i])
        end
    end

    return C
end

end