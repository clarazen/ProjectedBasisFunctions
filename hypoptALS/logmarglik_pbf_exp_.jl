#module logmarklik

#using LinearAlgebra
#using ..functionsALSmodelweights
#using ..functionsTT

#export logmarglik_pbf_exp

#function logmarglik_pbf_exp(hyp::Vector{Float64},X::Matrix{Float64},y::Vector{Float64},Φ::Vector{Matrix},tt::TTv,M::Vector{Float64},L::Vector{Float64},dd::Int64)
function logmarglik_pbf_exp(hyp,X,y,Φ,tt,dd,hyp1,hyp2,hyp3)

    push!(hyp1,exp(hyp[1]))
    push!(hyp2,exp(hyp[2]))
    push!(hyp3,exp(hyp[3]))
    # update model weights and compute projected basis functions
    tt,Φp,res       = ALS_modelweights(X,y,Φ,exp.(hyp),dd,tt);
    # compute log marginal likelihood terms
    N               = size(X,1)
    Z               = I + 1/σ_n²*Φp'*Φp;
    Lchol           = cholesky(Z).L;
    term1           = sum(log.(diag(Lchol))) + N/2*log(σ_n²) 
    term2           = 1/2σ_n² * y'*y - 1/2σ_n²^2 * y'*Φp*(Z\(Φp'*y))
    term3           = N/2 * log(2π)

    return  term1 + term2 + term3

end

function logmarglik_pbf_exp(hyp,X,y,Φ,tt,dd,hyp1,hyp2,hyp3,err,Φstar_,ftest)

    push!(hyp1,exp(hyp[1]))
    push!(hyp2,exp(hyp[2]))
    push!(hyp3,exp(hyp[3]))
    # update model weights and compute projected basis functions
    tt,Φp,res       = ALS_modelweights(X,y,Φ,exp.(hyp),dd,tt);

    # compute relative error
    mstar           = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
    push!(err,RMSE(mstar,ftest))

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