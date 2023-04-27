using DelimitedFiles
using StatsBase
using Revise
using LinearAlgebra
using Plots
using Optim
using Metrics

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
#includet("hypoptALS/logmarglik_pbf_exp_.jl")
#using .logmarklik

X                   = Xtotn[1:36584,1:9];
y                   = ytotn[1:36584];
Xstar               = Xtotn[36585:end,1:9];
ystar               = ytotn[36585:end];
D                   = 9;
Md                  = 30;
N                   = 36584;

dd                  = 5;
# find weights without prior
hyp0                = [1., 1., 1.];
ℓ²,σ_f²,σ_n²        = hyp0;
L                   = ones(D) .+ 2*sqrt(ℓ²);
M                   = 3*ones(D);
rnks                = Int.([1,3*ones(D-1)...,1]);
Φ,Λ                 = colectofbasisfunc(M,X,ℓ²,σ_f²,L,1);
tt,cov,res          = ALS_modelweights(y,Φ,rnks,10,0.0001,dd);

# optimize in subspace
hyp1                = [];
hyp2                = [];
hyp3                = [];
obj                 = hyp -> logmarglik_pbf_exp(hyp,X,y,Φ,tt,hyp1,hyp2,hyp3)
optres              = optimize(obj,log.(hyp0))
ℓ²,σ_f²,σ_n²        = exp.(Optim.minimizer(optres)) # optimized

# recompute weights with prior
Φstar_              = colectofbasisfunc(M,Xstar,ℓ²,σ_f²,L);
Φ_                  = colectofbasisfunc(M,X,ℓ²,σ_f²,L);
tt,cov,res          = ALS_modelweights(y,Φ_,10,σ_n²,dd,tt);

mstar               = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm                 = getttm(tt,dd);  
Φstarttm            = khrtimesttm(Φstar_,ttm);
P_tt                = Φstarttm*cov*Φstarttm';
s_tt                = sqrt.(diag(P_tt))

RMSE(mstar,ystar)
MSLL(mstar[:,1],s_tt,ystar,sqrt(σ_n²))
norm(mstar-ystar)/norm(ystar)

r2_score(mstar,ystar)

