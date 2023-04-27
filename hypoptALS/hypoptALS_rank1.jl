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

N                   = 500;     # number of data points
D                   = 3;        # dimensions
Md                  = 20;       # number of basis functions per dimension
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
L                   = ones(D) .+ 2*sqrt(ℓ²);

# learning and predictions with 'generative' hyperparameters
Φstar_              = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L);
Φ_                  = colectofbasisfunc(M,X,ℓ²,σ_f²,L);
tt,cov,res          = ALS_modelweights(y,Φ_,rnks,maxiter,σ_n²,2);

mstar               = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm                 = getttm(tt,2);  
Φstarttm            = khrtimesttm(Φstar_,ttm);
P_tt                = Φstarttm*cov*Φstarttm';
s_tt                = sqrt.(diag(P_tt))

RMSE(mstar,ftest)
MSLL(mstar[:,1],s_tt,ftest,sqrt(σ_n²))
norm(mstar-ftest)/norm(ftest)

###########################################################################################################
# optimize hyperparameters (full GP)
hyp                 = [ℓ²,σ_f²,σ_n²];
obj                 = hyp -> logmarglik_full_exp(hyp,X,y)
optres              = optimize(obj,log.(hyp),Optim.Options(iterations = 100))
ℓ²,σ_f²,σ_n²        = exp.(Optim.minimizer(optres)) # optimized

# optimize hyperparameters (Hilbert-GP)
hyp                 = [ℓ²,σ_f²,σ_n²];
obj                 = hyp -> logmarglik_bf_exp(hyp,X,y,1000)
optres              = optimize(obj,log.(hyp),Optim.Options(iterations = 100))
ℓ²,σ_f²,σ_n²        = exp.(Optim.minimizer(optres)) # optimized

# optimize hyperparameters (Hilbert-GP, rank-1 aprox)
hyp                 = [ℓ²,σ_f²,σ_n²];

ℓ²,σ_f²,σ_n²    = exp.(hyp) 
boundsMin       = minimum(X,dims=1);
boundsMax       = maximum(X,dims=1);
L               = ((boundsMax.-boundsMin) ./ 2)[1,:] .+ 2*ℓ²; 
Φ,Λ,ind         = colectofbasisfunc(M,X,ℓ²,σ_f²,L);

invΛ            = diagm(1 ./ diag(Λ));
Z               = σ_n²*invΛ + Φ'*Φ;
F               = svd(Z);
Zrank1          = F.U[:,1]*F.S[1]*F.Vt[1,:]'
invZrank1       = F.U[:,1]*(1/F.S[1])*F.Vt[1,:]'

term1           = 1/2 * ((N-1000) * log(σ_n²) + log(det(Zrank1)) + sum(log.(diag(Λ))))
term2           = 1/2σ_n² * (y'*y - (y'*Φ) *invZrank1* (Φ'*y) )
term3           = N/2 * log(2π)

obj                 = hyp -> logmarglik_bf_exp_rank1(hyp,X,y,1000)
optres              = optimize(obj,log.(hyp),Optim.Options(iterations = 100))
ℓ²,σ_f²,σ_n²        = exp.(Optim.minimizer(optres)) # optimized

