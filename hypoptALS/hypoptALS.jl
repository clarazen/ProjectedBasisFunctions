using LinearAlgebra
#using Revise

includet("../functions/functionsBasic.jl")
using .functionsBasic
includet("../functions/functions_KhatriRao_Kronecker.jl")
using .functions_KhatriRao_Kronecker
includet("../functions/functionsBasisFunctions.jl")
using .functionsBasisfunctions
includet("../functions/functionsMetrics.jl")
using .functionsMetrics
includet("../functions/functionsTT.jl")
using .functionsTT
includet("../functions/functionsALSmodelweights.jl")
using .functionsALSmodelweights
includet("../functions/functionsTTmatmul.jl")
using .functionsTTmatmul

N               = 5000;     # number of data points
D               = 3;        # dimensions
Md              = 20;       # number of basis functions per dimension
hyp             = [1, 1., 0.01];
X,y,f,K         = gensynthdata(N,D,hyp);

boundsMin       = minimum(X,dims=1);
boundsMax       = maximum(X,dims=1);
L               = ((boundsMax.-boundsMin) ./ 2)[1,:] .+ 2*hyp[1]; 

# test for term2
ΦR,ΛR           = colectofbasisfunc(8000,X,hyp[1],hyp[2],L);
Φ,Λ             = colectofbasisfunc(Md*ones(D),X,hyp[1],hyp[2],L,true);
sum(ΛR) - prod(sum.(Λ))

norm(K-ΦR*ΛR*ΦR')/norm(K)

logmarglik_bf(hyp,y,X,L,8000)
logmarglik_full(hyp,X,y)


function logmarglik_bf(hyp,y,X,L,budget)
    ℓ²      = hyp[1];
    σ_f²    = hyp[2];
    σ_n²    = hyp[3];

    Φ,Λ     = colectofbasisfunc(budget,X,ℓ²,σ_f²,L);
    
    N       = size(X,1);
    M       = size(Λ,1);

    Z       = σ_n²*diagm(1 ./ diag(Λ)) + Φ'*Φ;
    Lchol   = cholesky(Z).L
    term1   = (N-M) * log(σ_n²) + 2*sum(log.(diag(Lchol))) + sum(log.(diag(Λ)))
    term2   = 1/σ_n² * (y'*y - y'*Φ* (Lchol'\(Lchol\(Φ'*y))) )
    term3   = N * log(2π)

    return 1/2* (term1 + term2 + term3)

end

function logmarglik_full(hyp,X,y)
    ℓ²      = hyp[1];
    σ_f²    = hyp[2];
    σ_n²    = hyp[3];
    N       = size(X,1);

    K       = covSE(X,X,[ℓ²,σ_f²,σ_n²])
    L       = cholesky((K + σ_n²*Matrix(I,N,N))).L
    N       = size(X,1);

    α       = L'\(L\y)

    term1   = sum(log.(diag(L)))
    term2   = y'*α
    term3   = N * log(2π)

    return 1/2 * (term1 + term2 + term3)

end

function logmarglik_ALS(hyp,y,rnks,maxiter)
    ℓ²      = hyp[1];
    σ_f²    = hyp[2];
    σ_n²    = hyp[3];

    Φ,Λ     = colectofbasisfunc(Md*ones(D),Xall,ℓ²,σ_f²,L,true);
    Φ_      = colectofbasisfunc(Md*ones(D),Xall,ℓ²,σ_f²,L);
    
    N       = size(Φ_[1],1);
    M       = prod(size.(Φ_,2));

    logdetZ = 

    term1   = 1/σ_n² * (y'*y - y'*khrtimesttm(Φ_,ALS_modelweights(y,Φ_,rnks,maxiter,σ_n²)))
    term2   = (N-M)*log(σ_n²) + logdetZ + prod(sum.(Λ))
    term3   = N/2 * log(2π)

    return term1 + term2 + term3

end