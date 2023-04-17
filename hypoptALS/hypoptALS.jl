using LinearAlgebra
#using Revise
using Optim

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

N                   = 500;     # number of data points
D                   = 3;        # dimensions
Md                  = 30;       # number of basis functions per dimension
ℓ²,σ_f²,σ_n²        = [.5, 1., 0.5];
Xall,yall,fall,Kall = gensynthdata(N,D,[ℓ²,σ_f²,σ_n²]);
X                   = Xall[1:400,:];
Xtest               = Xall[401:end,:];
y                   = yall[1:400];
ftest               = fall[401:end];
# modified Hilbert-GP with projected basis functions
M                   = Md*ones(D);
rnks                = Int.([1,10*ones(D-1)...,1]);
maxiter             = 5;
ℓ²,σ_f²,σ_n²        = [1, 1, 1.]; # initial guess for hyper parameters
# compute basis functions per dimension
L                   = ones(D) .+ 2*sqrt(ℓ²);
@run Φ_                  = colectofbasisfunc(M,X,ℓ²,σ_f²,L);
Φstar_              = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L);

tt,cov,res          = ALS_modelweights(y,Φ_,rnks,maxiter,σ_n²,2);
Wd                  = getttm(tt,2);  

mstar               = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm                 = getttm(tt,2);  
Φstarttm            = khrtimesttm(Φstar_,ttm);
P_tt                = Φstarttm*cov*Φstarttm';
s_tt                = sqrt.(diag(P_tt))

RMSE(mstar,ftest)
MSLL(mstar[:,1],s_tt,ftest,sqrt(σ_n²))
norm(mstar-ftest)/norm(ftest)

@run logmarglik          = logmarglik_pbf_exp_(log.([ℓ²,σ_f²,σ_n²]),X,y,M,Wd)
ℓ²,σ_f²,σ_n²        = exp.(Optim.minimizer(optimize(logmarglik_pbf_exp,log.([ℓ²,σ_f²,σ_n²]))))
logmarglik          = logmarglik_pbf_exp_(log.([ℓ²,σ_f²,σ_n²]),X,y,M,Wd)

ℓ²,σ_f²,σ_n²        = [1, 1, 1.];
logmarglik          = logmarglik_full_exp_(log.([ℓ²,σ_f²,σ_n²]),X,y)
ℓ²,σ_f²,σ_n²        = Optim.minimizer(optimize(logmarglik_full_exp,log.([ℓ²,σ_f²,σ_n²])))
logmarglik          = logmarglik_full_exp_([ℓ²,σ_f²,σ_n²],X,y)

