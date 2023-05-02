using LinearAlgebra
using Revise
using Optim
using Plots
using DelimitedFiles
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

X                   = Xtotn[1:36584,1:9];
y                   = ytotn[1:36584];
Xstar               = Xtotn[36585:end,1:9];
ystar               = ytotn[36585:end];
D                   = 9;
Md                  = 30;
N                   = 36584;

dd                  = 4;

ρ                   = 3
knotint             = 1
M                   = Int.((ρ + knotint)*ones(D))

maxiter             = 2;
Φ                   = bsplines(X,ρ,knotint);
Φstar               = bsplines(Xstar,ρ,knotint);

rnks                = Int.([1,4,10*ones(D-3)...,4,1]);
@time tt,res,ΦWd              = ALS_modelweights(y,Φ,rnks,maxiter,0.0,ρ,knotint,dd);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar)
norm(mstar-ystar)/norm(ystar)

# compute penalty matrix
P       = diff(I(ρ + knotint),dims=1);
PP      = P'*P;
Wpen    = penalmat(tt,dd,D,P,PP)
# Weighted sum of the difference penalty matrices
λ = 1e-10
WWW = λ*Wpen[1];
for i =2:D
    WWW = WWW + λ*Wpen[i];
end
tt[dd]      = reshape(pinv(ΦWd'*ΦWd + λ*WWW)*(ΦWd'*y),size(tt[dd]))
norm(tt[dd])
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar)

tt,res,ΦWd              = ALS_modelweights(y,Φ,rnks,maxiter,λ,ρ,knotint,dd);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar)
norm(mstar-ystar)/norm(ystar)

r2_score(mstar,ystar)