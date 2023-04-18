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
ℓ²,σ_f²,σ_n²        = [1, 1, 1.]; # initial guess for hyper parameters
# compute basis functions per dimension
L                   = ones(D) .+ 2*sqrt(ℓ²);
Φ,Λ                 = colectofbasisfunc(M,X,ℓ²,σ_f²,L,1);

# create site-d canonical initial tensor train    
dd                 = 2
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

hyp                 = [.5, 1., 0.5];
logmarglik_pbf_exp(hyp,X,y,Φ,tt,2,[],[],[])

hyp1                = []
hyp2                = []
hyp3                = []
obj                 = hyp -> logmarglik_pbf_exp(hyp,X,y,Φ,tt,2,hyp1,hyp2,hyp3)
optres              = optimize(obj,log.(hyp),Optim.Options(iterations = 100))

############## computing error
hyp                 = [1., 2., 0.1];
Φstar_              = colectofbasisfunc(M,Xtest,ℓ²,σ_f²,L);
hyp1                = []
hyp2                = []
hyp3                = []
err                 = []
obj                 = hyp -> logmarglik_pbf_exp(hyp,X,y,Φ,tt,2,hyp1,hyp2,hyp3,err,Φstar_,ftest)
optres              = optimize(obj,log.(hyp),Optim.Options(iterations = 50))
##############
ℓ²,σ_f²,σ_n²        = exp.(Optim.minimizer(optres))
logmarglik_pbf_exp(hyp,X,y,Φ,tt,2,hyp1,hyp2,hyp3)

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

plot(hyp1)
plot(hyp2)
plot(hyp3)
