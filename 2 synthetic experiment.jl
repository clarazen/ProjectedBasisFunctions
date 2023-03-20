using LinearAlgebra
using Revise

includet("functions/functionsBasic.jl")
using .functionsBasic
includet("functions/functions_KhatriRao_Kronecker.jl")
using .functions_KhatriRao_Kronecker
includet("functions/functionsBasisFunctions.jl")
using .functionsBasisfunctions
includet("functions/functionsMetrics.jl")
using .functionsMetrics
includet("functions/functionsTT.jl")
using .functionsTT
includet("functions/functionsALSmodelweights.jl")
using .functionsALSmodelweights
includet("functions/functionsTTmatmul.jl")
using .functionsTTmatmul

N               = 5000;     # number of data points
D               = 3;        # dimensions
Md              = 20;       # number of basis functions per dimension
hyp             = [0.05, 1., 0.001];
Xall,~,~,K_tot  = gensynthdata(N,D,hyp);

boundsMin       = minimum(Xall,dims=1);
boundsMax       = maximum(Xall,dims=1);
L               = 1.5*((boundsMax.-boundsMin) ./ 2)[1,:]; 

Φ_tot           = colectofbasisfunc(Md*ones(D),Xall,hyp[1],hyp[2],L);
Φ_tot_mat       = khr2mat(Φ_tot);

# kernel matrix approximation
K̃_tot           = Φ_tot_mat*Φ_tot_mat';
                norm(K_tot-K̃_tot)/norm(K_tot)

R               = 5#1, 5,10,20
w1              = Matrix(qr(randn(Md,R)).Q);
w2              = randn(R*Md*R);
w3              = Matrix(qr(randn(Md,R)).Q);
Wd              = kron(kron(w3,Matrix(I,Md,Md)),w1);

fall            = Φ_tot_mat*Wd*w2;
SNR             = 10;
noise_norm      = norm(fall)/(10^(SNR/20));
σ_n             = sqrt(noise_norm^2/(length(fall)-1));
noise           = randn(size(fall));
noise           = noise/norm(noise)*noise_norm;
yall            = fall + noise;
# training and validation data
X               = Xall[1:4000,:];
Xstar           = Xall[4001:5000,:];
y               = yall[1:4000];
ystar           = fall[4001:5000];
    

hyp[3]          = σ_n^2
Φ_              = colectofbasisfunc(Md*ones(D),X,hyp[1],hyp[2],L);
Φ_mat           = khr2mat(Φ_);
Φstar_          = colectofbasisfunc(Md*ones(D),Xstar,hyp[1],hyp[2],L);
Φstar_mat       = khr2mat(Φstar_);

# full GP
K               = covSE(X,X,hyp);
mstar,vstar     = fullGP(K,X,Xstar,y,hyp);
MSLL_gp         = MSLL(mstar,vstar,ystar,σ_n)
SMSE_gp         = SMSE(mstar,ystar,y)
RMSE_gp         = RMSE(mstar,ystar)
                norm(ystar-mstar)/norm(ystar)

# tt approximation
# ALS starting and finishing in second core
rnks            = Int.([1, R*ones(D-1,1)..., 1]);
maxiter         = 5;
tt,cov2,res     = ALS_modelweights(y,Φ_,rnks,maxiter,hyp[3],2);
m_tt2           = khrtimesttm(Φstar_,tt2ttm(tt,Int.(vcat(Md',ones(D)'))))[:,1];
ttm             = getttm(tt,2);  
Φstarttm        = khrtimesttm(Φstar_,ttm);
P_tt2           = hyp[3]*Φstarttm*cov2*Φstarttm';
P_tt2           = diag(P_tt2);
MSLL_tt         = MSLL(m_tt2[:,1],P_tt2,ystar,σ_n)
SMSE_tt         = SMSE(m_tt2[:,1],ystar,y)
RMSE_tt         = RMSE(m_tt2[:,1],ystar)
                norm(ystar-m_tt2)/norm(ystar)

## rr approximation
budget          = R*R*Md
ΦR,ΛR           = colectofbasisfunc(budget,X,hyp[1],hyp[2],L);
ΦsR,ΛsR         = colectofbasisfunc(budget,Xstar,hyp[1],hyp[2],L);

K̃               = ΦR*ΛR*ΦR';
                norm(K-K̃)/norm(K)

tmp             = ΦR'*ΦR + (hyp[3]*diagm(1 ./ diag(ΛR)));
w_rr            = tmp\(ΦR'*y);
m_rr            = ΦsR * w_rr;
P_rr            = hyp[3]*ΦsR * inv(tmp) * ΦsR';
P_rr            = diag(P_rr);
MSLL_rr         = MSLL(m_rr[:,1],P_rr,ystar,σ_n)
SMSE_rr         = SMSE(m_rr[:,1],ystar,y)
RMSE_rr         = RMSE(m_rr[:,1],ystar)
                norm(ystar-m_rr)/norm(ystar)


