using LinearAlgebra
using Revise
using Optim
using Plots
using DelimitedFiles

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

X                   = readdlm("hypoptALS/X_cascaded.csv",',');
Xtest               = readdlm("hypoptALS/Xtest_cascaded.csv",',');
y                   = readdlm("hypoptALS/y_cascaded.csv",',')[:,1];
ystar               = readdlm("hypoptALS/ytest_cascaded.csv",',')[:,1];
N,D                 = size(X)

ρ                   = 3
knotint             = 1
M                   = Int.((ρ + knotint)*ones(D))

rnks                = Int.([1,4,10*ones(D-3)...,4,1]);
maxiter             = 2;
Φ                   = bsplines(X,ρ,knotint);
Φstar               = bsplines(Xtest,ρ,knotint);
#@time tt,res              = ALS_modelweights(y,Φ,rnks,maxiter,1.9307e-07,ρ,knotint,8);
#res[end]
#mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
#RMSE(mstar,ystar[33:end])
#norm(mstar-ystar[33:end])/norm(ystar[33:end])
#plot(ystar[33:end])
#plot!(mstar)


@time tt,res,ΦWd              = ALS_modelweights(y,Φ,rnks,maxiter,0.0,ρ,knotint,8);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar[33:end])
norm(mstar-ystar[33:end])/norm(ystar[33:end])
plot(ystar[33:end])
plot!(mstar)
ylims!((0,1.5))

# compute penalty matrix
P       = diff(I(ρ + knotint),dims=1);
PP      = P'*P;
Wpen    = penalmat(tt,8,D,P,PP)
# Weighted sum of the difference penalty matrices
λ = 1e-12
WWW = λ*Wpen[1];
for i =2:D
    WWW = WWW + λ*Wpen[i];
end
tt[8]      = reshape(pinv(ΦWd'*ΦWd + λ*WWW)*(ΦWd'*y),size(tt[8]))
norm(tt[8])
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar[33:end])

tt,res,ΦWd              = ALS_modelweights(y,Φ,rnks,maxiter,λ,ρ,knotint,8);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar[33:end])
norm(mstar-ystar[33:end])/norm(ystar[33:end])
plot(ystar[33:end])
plot!(mstar)