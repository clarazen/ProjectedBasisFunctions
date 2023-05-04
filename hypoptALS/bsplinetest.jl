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
maxiter             = 10;
Φ                   = bsplines(X,ρ,knotint);
Φstar               = bsplines(Xtest,ρ,knotint);
#@time tt,res              = ALS_modelweights(y,Φ,rnks,maxiter,1.9307e-07,ρ,knotint,8);
#res[end]
#mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
#RMSE(mstar,ystar[33:end])
#norm(mstar-ystar[33:end])/norm(ystar[33:end])
#plot(ystar[33:end])
#plot!(mstar)

tt0                                 = initTT(rnks,M[1],3,D)
@time tt,covdd,res,ΦWd              = ALS_modelweights(y,Φ,maxiter,0.0,0.1,tt);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar[33:end])
plot(ystar[33:end])
plot!(mstar)
ylims!((0,1.5))

# compute penalty matrix
P       = diff(I(ρ + knotint),dims=1);
PP      = P'*P;
Wpen    = penalmat(tt,8,D,P,PP)
# Weighted sum of the difference penalty matrices

λ           = 1e-24
WWW = λ*Wpen[1];
for i =2:D
    WWW = WWW + λ*Wpen[i];
end

tt[8]      = reshape((ΦWd'*ΦWd + WWW)\(ΦWd'*y),size(tt[8]))
norm(y - ΦWd*tt[8][:])/norm(y)
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar[33:end])
plot(ystar[33:end])
plot!(mstar)

##################
# another ALS, now regularized (does not work with λ for particualr subspace)
tt,res,ΦWd              = ALS_modelweights(y,Φ,tt,maxiter,λ,ρ,knotint,8);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ystar[33:end])
norm(mstar-ystar[33:end])/norm(ystar[33:end])
plot(ystar[33:end])
plot!(mstar)



evol                = []
obj                 = hyp -> logmarglik_pbf_exp(hyp,y,Φ,tt,evol)
optres              = optimize(obj,[log(1)],LBFGS())
λ                   = exp.(Optim.minimizer(optres))[1] # optimized
plot(evol,yaxis=:log)


evol                = []
obj                 = hyp -> GCV(hyp,y,ΦWd,evol)


plot([GCV(1e-23,y,ΦWd,evol),GCV(1e-24,y,ΦWd,evol),GCV(1e-25,y,ΦWd,evol),GCV(1e-30,y,ΦWd,evol)])

optres              = optimize(obj,[1e-25],LBFGS())
λ                   = Optim.minimizer(optres)[1]
plot(evol,yaxis=:log)