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
D               = 4;        # dimensions
Md              = 10;       # number of basis functions per dimension
hyp             = [0.2, 1., 0.001];
Xall,~,~,K_tot  = gensynthdata(N,D,hyp);

boundsMin       = minimum(Xall,dims=1);
boundsMax       = maximum(Xall,dims=1);
L               = 1.5*((boundsMax.-boundsMin) ./ 2)[1,:]; 

Φ_tot           = colectofbasisfunc(Md*ones(D),Xall,hyp[1],hyp[2],L);
Φ_tot_mat       = khr2mat(Φ_tot);

# kernel matrix approximation
K̃_tot           = Φ_tot_mat*Φ_tot_mat';
                norm(K_tot-K̃_tot)/norm(K_tot)

R               = 10#1, 5,10,20
tt              = TT([randn(1,Md,R),randn(R,Md,R^2),randn(R^2,Md,R),randn(R,Md,1)]);
                shiftTTnorm(tt,1,1);shiftTTnorm(tt,2,1);shiftTTnorm(tt,3,1);shiftTTnorm(tt,4,-1);
W3              = ttm2mat(getttm(tt,3));
w3              = randn(R^2*Md*R);

fall            = Φ_tot_mat*W3*w3;
SNR             = 20;
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
# ALS to find weight matrix
rnks            = Int.([1, R,R^2,R, 1]);
maxiter         = 5;
tt,res          = ALS_modelweights(y,Φ_,rnks,maxiter,hyp[3]);
# Bayesian update of second core
                shiftTTnorm(tt,D,-1);
ttm             = getttm(tt,3);  

# this crashes
U               = khrtimesttm(Φ_,ttm); 
tmp             = U'*U;
tmp             = tmp + hyp[3]*Matrix(I,size(tmp));     

tt[3]           = reshape(tmp\(U'*y),size(tt[3]));
m_tt            = Φstar_mat*ttv2vec(tt);
cova2           = inv(tmp);
Φstarttm        = khrtimesttm(Φstar_,ttm);
P_tt            = hyp[3]*Φstarttm*cova2*Φstarttm';
P_tt            = diag(P_tt);
MSLL_tt         = MSLL(m_tt[:,1],P_tt,ystar,σ_n)
SMSE_tt         = SMSE(m_tt[:,1],ystar,y)
RMSE_tt         = RMSE(m_tt[:,1],ystar)
                norm(ystar-m_tt)/norm(ystar)

## rr approximation
budget          = R*R^2*Md
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


