using LinearAlgebra
using Revise
using Optim
using Plots

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

N                   = 2000;     # number of data points
D                   = 3;        # dimensions
ℓ²,σ_f²,σ_n²        = [.5, 1., 0.01];
ρ                   = 3
knotint             = 1
M                   = Int.((ρ + knotint)*ones(D));
Xall                = zeros(N,D)
for d = 1:D
    for n = 1:N
        Xall[n,d] = rand(1)[1].* 2 .-1
    end
end
Φall                = bsplines(Xall,ρ,knotint);
Wd                  = Matrix(qr(randn(64,64)));
fall                = khr2mat(Φall)*Wd*rand(64)
yall                = fall + sqrt(σ_n²)*randn(N)

X                   = Xall[1:1800,:];
Xtest               = Xall[1801:end,:];
y                   = yall[1:1800];
ftest               = fall[1801:end];



rnks                = Int.([1,4,4,1]);
maxiter             = 10;
Φ                   = bsplines(X,ρ,knotint);
Φstar               = bsplines(Xtest,ρ,knotint);

@time tt,res,ΦWd    = ALS_modelweights(y,Φ,rnks,maxiter,0.0,ρ,knotint,2);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
RMSE(mstar,ftest)

###########################################################################################################
# optimize hyperparameters in subspace found with NON Bayesian ALS
# compute penalty matrix
dd                  = 2
P                   = diff(I(ρ + knotint),dims=1);
PP                  = P'*P;
Wpen                = penalmat(tt,dd,D,P,PP)
# Weighted sum of the difference penalty matrices
λ                   = 1e-2
WWW                 = λ*Wpen[1];
for i = 2:D
    WWW             = WWW + λ*Wpen[i];
end
tt[dd]              = reshape(pinv(ΦWd'*ΦWd + WWW)*(ΦWd'*y),size(tt[dd]))
                    norm(y - ΦWd*tt[dd][:])/norm(y)
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
                    RMSE(mstar,ftest)

