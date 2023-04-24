using DelimitedFiles
using StatsBase
using Revise
using LinearAlgebra
using Plots
using Optim

includet("functions/functionsBasic.jl")
using .functionsBasic
includet("functions/functions_KhatriRao_Kronecker.jl")
using .functions_KhatriRao_Kronecker
includet("functions/functionsBasisFunctions.jl")
using .functionsBasisfunctions
includet("functions/functionsMetrics.jl")
using .functionsMetrics
includet("functions/functionsTT.jl")
using .functionsTT
includet("functions/functionsALSmodelweights.jl")
using .functionsALSmodelweights
includet("functions/functionsTTmatmul.jl")
using .functionsTTmatmul
#includet("hypoptALS/logmarglik_pbf_exp_.jl")
#using .logmarklik

X               = Matrix(readdlm("../ProjectedBasisFunctions - benchmark data/X_kukainv.csv",','))
y               = Matrix(readdlm("../ProjectedBasisFunctions - benchmark data/y_kukainv.csv",','))[:,1]
Xtest           = Matrix(readdlm("../ProjectedBasisFunctions - benchmark data/Xtest_kukainv.csv",','))
ytest           = Matrix(readdlm("../ProjectedBasisFunctions - benchmark data/ytest_kukainv.csv",','))[:,1]

# ℓ²,σ_f²,σ_n²    = [.4706^2,2.8853^2,0.6200^2]; # hyperparameters optimized with full GP
D               = size(X,2);
dd              = 8;
# find weights without prior
hyp0            = [1., 1., 1.];
ℓ²,σ_f²,σ_n²    = hyp0;
L               = ones(D) .+ 2*sqrt(ℓ²);
M               = 30*ones(D);
rnks            = Int.([1,3*ones(D-1)...,1]);
Φ,Λ             = colectofbasisfunc(M,X,ℓ²,σ_f²,L,1);
@time tt,cov,res,ΦWd      = ALS_modelweights(y,Φ,rnks,10,0.0001,dd);

# optimize in subspace
hyp1                = [];
hyp2                = [];
hyp3                = [];
obj                 = hyp -> logmarglik_pbf_exp(hyp,X,y,Φ,tt,hyp1,hyp2,hyp3)
optres              = optimize(obj,log.(hyp0))
ℓ²,σ_f²,σ_n²        = exp.(Optim.minimizer(optres)) # optimized

# recompute weights with prior
Φstar_              = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L);
Φ_                  = colectofbasisfunc(M,X,ℓ²,σ_f²,L);
# take last tt as initial guess
tt,cov,res          = ALS_modelweights(y,Φ_,rnks,10,σ_n²,dd);

mstar               = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm                 = getttm(tt,dd);  
Φstarttm            = khrtimesttm(Φstar_,ttm);
P_tt                = Φstarttm*cov*Φstarttm';
s_tt                = sqrt.(diag(P_tt))

RMSE(mstar,ytest)
MSLL(mstar[:,1],s_tt,ytest,sqrt(σ_n²))
norm(mstar-ytest)/norm(ytest)

plot(ytest)
plot!(mstar,ribbon=[s_tt s_tt])


