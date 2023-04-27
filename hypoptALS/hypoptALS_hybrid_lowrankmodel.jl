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

N                   = 5000;     # number of data points
D                   = 3;        # dimensions
Md                  = 10;       # number of basis functions per dimension
ℓ²,σ_f²,σ_n²        = [.5, 1., 0.5];
M                   = Md*ones(D);

Xall,~,~,K_tot      = gensynthdata(N,D,[ℓ²,σ_f²,σ_n²]);

boundsMin           = minimum(Xall,dims=1);
boundsMax           = maximum(Xall,dims=1);
L                   = 1.5*((boundsMax.-boundsMin) ./ 2)[1,:]; 

Φ_tot               = colectofbasisfunc(Md*ones(D),Xall,ℓ²,σ_f²,L);
Φ_tot_mat           = khr2mat(Φ_tot)

R                   = 10
w1                  = Matrix(qr(randn(Md,R)).Q);
w2                  = randn(R*Md*R);
w3                  = Matrix(qr(randn(Md,R)).Q);
Wd                  = kron(kron(w3,Matrix(I,Md,Md)),w1);

fall                = Φ_tot_mat*Wd*w2;
SNR                 = 10;
noise_norm          = norm(fall)/(10^(SNR/20));
σ_n                 = sqrt(noise_norm^2/(length(fall)-1));
noise               = randn(size(fall));
noise               = noise/norm(noise)*noise_norm;
yall                = fall + noise;
# training and validation data
X                   = Xall[1:4000,:];
Xtest               = Xall[4001:5000,:];
y                   = yall[1:4000];
ftest               = fall[4001:5000];

###########################################################################################################
# optimize hyperparameters in subspace found with NON Bayesian ALS
rnks                = [1,10,10,1];
Φ,Λ                 = colectofbasisfunc(M,X,ℓ²,σ_f²,L,1);
Φstar,Λstar         = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L,1);
tt,cov,res          = ALS_modelweights(y,Φ,rnks,10,0.0,2);
res[end]
mstar               = khrtimesttm(Φstar,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm                 = getttm(tt,2);  
Φstarttm            = khrtimesttm(Φstar,ttm);
P_tt                = Φstarttm*cov*Φstarttm';
s_tt                = sqrt.(diag(P_tt))
RMSE(mstar,ftest) # training error for weights without any prior
MSLL(mstar[:,1],s_tt,ftest,sqrt(σ_n²))
norm(mstar-ftest)/norm(ftest)

# optimization
hyp1                = []
hyp2                = []
hyp3                = []
obj                 = hyp -> logmarglik_pbf_exp(hyp,X,y,Φ,tt,hyp1,hyp2,hyp3)
optres              = optimize(obj,log.([.5, 1., 0.5]), LBFGS())
ℓ²,σ_f²,σ_n²        = exp.(Optim.minimizer(optres)) # optimized
p1 = plot(hyp1); p2 = plot(hyp2); p3 = plot(hyp3);
plot(p1,p2,p3,layout=(3,1))

Φstar_              = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L);
Φ_                  = colectofbasisfunc(M,X,ℓ²,σ_f²,L);
tt,cov,res          = ALS_modelweights(y,Φ_,rnks,10,σ_n²,2);

mstar               = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm                 = getttm(tt,2);  
Φstarttm            = khrtimesttm(Φstar_,ttm);
P_tt                = Φstarttm*cov*Φstarttm';
s_tt                = sqrt.(diag(P_tt))

RMSE(mstar,ftest)
MSLL(mstar[:,1],s_tt,ftest,sqrt(σ_n²))
norm(mstar-ftest)/norm(ftest)

