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

X                   = Xtotn[1:32000,:];
y                   = ytotn[1:32000];
Xstar               = Xtotn[32001:end,:];
ystar               = ytotn[32001:end];
D                   = 8;
Md                  = 10;
N                   = 32000;

dd                  = 5;
# find weights without prior
hyp0                = [1., 1., 1.];
ℓ²,σ_f²,σ_n²        = hyp0;
L                   = ones(D) .+ 2*sqrt(ℓ²);
M                   = 5*ones(D);
rnks                = Int.([1,5*ones(D-1)...,1]);
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
left,right          = initsupercores(Φstar_,tt,dd);
Φstarttm            = KhRxTTm(dd,left[dd-1],right[dd+1],Φstar_[dd],D);
P_tt                = Φstarttm*cov*Φstarttm';
s_tt                = sqrt.(diag(P_tt))

RMSE(mstar,ystar)
MSLL(mstar[:,1],s_tt,ystar,sqrt(σ_n²))
norm(mstar-ystar)/norm(ystar)

r2_score(mstar,ystar)

@run Φt,Λt                  = colectofbasisfunc(M,X,ℓ²,σ_f²,L,1);