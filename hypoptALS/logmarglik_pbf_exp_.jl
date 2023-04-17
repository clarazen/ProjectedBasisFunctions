#module logmarklik

#using LinearAlgebra
#using ..functionsALSmodelweights
#using ..functionsTT

#export logmarglik_pbf_exp

#function logmarglik_pbf_exp(hyp::Vector{Float64},X::Matrix{Float64},y::Vector{Float64},Φ::Vector{Matrix},tt::TTv,M::Vector{Float64},L::Vector{Float64},dd::Int64)
function logmarglik_pbf_exp(hyp,X,y,Φ,tt,M,L,dd)
    ℓ²,σ_f²,σ_n²    = exp.(hyp) 
    D               = size(Φ,1)
    # compute √Λ from hyp
    sqrtΛ           = Vector{Vector}(undef,D);
    for d = 1:D
        w           = collect(1:M[d])';
        sqrtΛ[d]    = σ_f²^(1/D)*sqrt(2π*ℓ²) .* exp.(- ℓ²/2 .* ((π.*w')./(2L[d])).^2 )
    end
    # compute √Λ*Φ
    Φ_              = Vector{Matrix}(undef,D)
    for d = 1:D
        Φ_[d]       = Φ[d]*diagm(sqrtΛ[d])
    end
    # compute projected basis functions
    left,right      = initsupercores(Φ_,tt,dd);
    Φp              = KhRxTTm(dd,left[dd-1],right[dd+1],Φ_[dd],size(Φ,1));

    # compute log marginal likelihood terms
    N               = size(X,1)
    Z               = I + 1/σ_n²*Φp'*Φp;
    Lchol           = cholesky(Z).L;
    term1           = sum(log.(diag(Lchol))) + N/2*log(σ_n²) 
    term2           = 1/2σ_n² * y'*y - 1/2σ_n²^2 * y'*Φp*(Z\(Φp'*y))
    term3           = N/2 * log(2π)

    return  term1 + term2 + term3

end


# is it possible to update a variable in side this function that is an input to it?

#end