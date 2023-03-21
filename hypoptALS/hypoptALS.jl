using LinearAlgebra
using Revise

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
hyp             = [0.05, 1., 0.001];
Xall,~,~,K_tot  = gensynthdata(N,D,hyp);

boundsMin       = minimum(Xall,dims=1);
boundsMax       = maximum(Xall,dims=1);
L               = 1.5*((boundsMax.-boundsMin) ./ 2)[1,:]; 

# test for term2
ΦR,ΛR           = colectofbasisfunc(8000,Xall,hyp[1],hyp[2],L);
Φ,Λ             = colectofbasisfunc(Md*ones(D),Xall,hyp[1],hyp[2],L,true);
sum(ΛR) - prod(sum.(Λ))


function logmarglik(hyp,y,rnks,maxiter)
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