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
L                   = ones(D) .+ 2*sqrt(ℓ²);

# learning and predictions with 'generative' hyperparameters
Φstar_              = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L);
Φ_                  = colectofbasisfunc(M,X,ℓ²,σ_f²,L);
tt,cov,res          = ALS_modelweights(y,Φ_,rnks,maxiter,0.5,2);

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

###########################################################################################################
# optimize hyperparameters (projected basis functions - attempt 1)
ℓ²,σ_f²,σ_n²        = [1., 1, 1]; # initial guess for hyper parameters
Φ,Λ                 = colectofbasisfunc(M,X,ℓ²,σ_f²,L,1);
Φstar,Λstar         = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L,1);
# create site-dd canonical initial tensor train    
dd                 = 2;
cores              = Vector{Array{Float64,3}}(undef,D);
for d = 1:dd-1 
    tmp            = qr(rand(rnks[d]*size(Φ[d],2), rnks[d+1]));
    cores[d]       = reshape(Matrix(tmp.Q),(rnks[d], size(Φ[d],2), rnks[d+1]));
end
cores[dd]          = reshape(rand(rnks[dd]*size(Φ[dd],2)*rnks[dd+1]),(rnks[dd], size(Φ[dd],2), rnks[dd+1]))
for d = dd+1:D
    tmp            = qr(rand(size(Φ[d],2)*rnks[d+1],rnks[d]));
    cores[d]       = reshape(Matrix(tmp.Q)',(rnks[d], size(Φ[d],2), rnks[d+1]));
end
tt                 = TT(cores,dd);

hyp                 = [ℓ²,σ_f²,σ_n²];
hyp1                = []
hyp2                = []
hyp3                = []
err                 = []
res                 = []
obj                 = hyp -> logmarglik_pbf_exp(hyp,X,y,Φ,tt,2,hyp1,hyp2,hyp3,err,res,Φstar,ftest)
optres              = optimize(obj,log.(hyp),Optim.Options(iterations = 100))
ℓ²,σ_f²,σ_n²        = exp.(Optim.minimizer(optres)) # optimized

# compute weights and make predictions with optimzied hyperparameters
Φ_                  = colectofbasisfunc(M,X,ℓ²,σ_f²,L);
Φstar_              = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L);

tt,cov,res_         = ALS_modelweights(y,Φ_,rnks,maxiter,σ_n²,2);

mstar               = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm                 = getttm(tt,2);  
Φstarttm            = khrtimesttm(Φstar_,ttm);
P_tt                = Φstarttm*cov*Φstarttm';
s_tt                = sqrt.(diag(P_tt))

RMSE(mstar,ftest)
MSLL(mstar[:,1],s_tt,ftest,sqrt(σ_n²))
norm(mstar-ftest)/norm(ftest)

plot(hyp1)
plot(hyp2)
plot(hyp3)

###########################################################################################################
# optimize hyperparameters (projected basis functions - attempt 2)
Φ,Λ                 = colectofbasisfunc(M,X,ℓ²,σ_f²,L,1);
Φstar,Λstar         = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L,1);
# create site-dd canonical initial tensor train    
dd                  = 2;
cores               = Vector{Array{Float64,3}}(undef,D);
for d = 1:dd-1 
    tmp             = qr(rand(rnks[d]*size(Φ[d],2), rnks[d+1]));
    cores[d]        = reshape(Matrix(tmp.Q),(rnks[d], size(Φ[d],2), rnks[d+1]));
end
cores[dd]           = reshape(rand(rnks[dd]*size(Φ[dd],2)*rnks[dd+1]),(rnks[dd], size(Φ[dd],2), rnks[dd+1]))
for d = dd+1:D
    tmp             = qr(rand(size(Φ[d],2)*rnks[d+1],rnks[d]));
    cores[d]        = reshape(Matrix(tmp.Q)',(rnks[d], size(Φ[d],2), rnks[d+1]));
end
tt                  = TT(cores,dd);

ℓ²,σ_f²,σ_n²        = [1, 1, 1]; # initial guess for hyper parameters
hyp                 = [ℓ²,σ_f²,σ_n²];
hyp1                = []
hyp2                = []
hyp3                = []
err                 = []
res                 = []
obj                 = hyp -> logmarglik_pbf_exp(hyp,X,y,Φ,tt,hyp1,hyp2,hyp3,err,res,Φstar,ftest)
optres              = optimize(obj,log.(hyp),Optim.Options(iterations = 100))
ℓ²,σ_f²,σ_n²        = exp.(Optim.minimizer(optres)) # optimized

# compute weights and make predictions with optimzied hyperparameters
Φ_                  = colectofbasisfunc(M,X,ℓ²,σ_f²,L);
Φstar_              = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L);

tt,cov,res          = ALS_modelweights(y,Φ_,rnks,maxiter,σ_n²,2);

mstar               = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(M',ones(D)'))))[:,1];
ttm                 = getttm(tt,2);  
Φstarttm            = khrtimesttm(Φstar_,ttm);
P_tt                = Φstarttm*cov*Φstarttm';
s_tt                = sqrt.(diag(P_tt))

RMSE(mstar,ftest)
MSLL(mstar[:,1],s_tt,ftest,sqrt(σ_n²))
norm(mstar-ftest)/norm(ftest)

plot(hyp1)
plot(hyp2)
plot(hyp3)