using LinearAlgebra
using Revise
using Pkg
Pkg.activate("C:/Users/cmmenzen/.julia/dev/BigMat")
using BigMat
Pkg.activate("C:/Users/cmmenzen/.julia/dev/TN4GP/")
using TN4GP
using StatsBase
using GaussianProcesses


N               = 5000;     # number of data points
D               = 3;        # dimensions
Md               = 40;       # number of basis functions per dimension
hyp             = [0.02*ones(D), 1., 0.001];
Xall,~,~,~      = gensynthdata(N,D,hyp);

boundsMin       = minimum(Xall,dims=1);
boundsMax       = maximum(Xall,dims=1);
L               = 1.5*((boundsMax.-boundsMin) ./ 2)[1,:]; 

Φ_              = colectofbasisfunc(Md*ones(D),Xall,hyp[1],hyp[2],L,true);
Φ_mat           = khr2mat(Φ_);
#kernel matrix approximation
K̃               = Φ_mat*Φ_mat';
K               = covSE(Xall,Xall,hyp);
                norm(K-K̃)/norm(K)

R               = 1 # 5,10,20
w1              = Matrix(qr(randn(Md,R)).Q);
w2              = randn(R*Md*R);
w3              = Matrix(qr(randn(Md,R)).Q);
Wd              = kron(kron(w3,Matrix(I,Md,Md)),w1)

fall            = Φ_mat*Wd*w2
SNR             = 10;
noise_norm      = norm(fall)/(10^(SNR/20))
σ_n             = sqrt(noise_norm^2/(length(fall)-1))
noise           = randn(size(fall))
noise           = noise/norm(noise)*noise_norm;
yall            = fall + noise
# training and validation data
X               = Xall[1:4000,:];
Xstar           = Xall[4001:5000,:];
y               = yall[1:4000];
ystar           = fall[4001:5000];

# hyperparameter optimization with GaussianProcesses toolbox    
mZero           = MeanZero()    ;               
kern            = SE([0.0,0.0,0.0],0.0) ;             
logObsNoise     = -1.0  ;                      
gp              = GP(Xall',yall,mZero,kern,logObsNoise);  
                optimize!(gp) 
hyp[1]          = 1 ./ gp.kernel.iℓ2
hyp[2]          = gp.kernel.σ2
hyp[3]          = exp(gp.logNoise.value)^2

Φ_              = colectofbasisfunc(Md*ones(D),X,hyp[1],hyp[2],L,true);
Φ_mat           = khr2mat(Φ_);
Φstar_          = colectofbasisfunc(Md*ones(D),Xstar,hyp[1],hyp[2],L,true);
Φstar_mat       = khr2mat(Φstar_);

# full GP
K               = covSE(X,X,hyp);
mstar,Pstar     = fullGP(K,X,Xstar,y,hyp,false);
                norm(ystar-mstar)/norm(ystar)
SMSE_gp,MSLL_gp = errormeasures(m_star,(P_star ./ 2).^2,ytest,σ_n)

# tt approximation
# ALS to find weight matrix
rnks            = Int.([1, R*ones(D-1,1)..., 1]);
maxiter         = 5;
tt,res          =  ALS_modelweights(y,Φ_,rnks,maxiter,hyp[3]);
# Bayesian update of second core
                shiftMPTnorm(tt,D,-1);
ttm             = getU(tt,2);   
U               = krtimesttm(Φ_,transpose(ttm)); 
tmp             = U*U';
tmp             = tmp + hyp[3]*Matrix(I,size(tmp));     
tt[2]           = reshape(tmp\(U*y),size(tt[2]))
m_tt            = Φstar_mat*mps2vec(tt);
cova2           = inv(tmp);
Φstarttm        = krtimesttm(Φstar_,transpose(ttm));
P_tt            = hyp[3]*Φstarttm'*cova2*Φstarttm;
P_tt            = diag(P_tt);
                norm(ystar-m_tt)/norm(ystar)
SMSE_tt,MSLL_tt = errormeasures(m_tt,P_tt,ytest,σ_n)
    

## rr approximation
budget          = R*R*Md;
budgetd         = Int.(ceil((budget)^(1/D)))
Φ,invΛ,S        = colectofbasisfunc(budgetd*ones(D),X,hyp[1],hyp[2],L);
Φstar,invΛs,~   = colectofbasisfunc(budgetd*ones(D),Xstar,hyp[1],hyp[2],L);
p               = sortperm(diag(mpo2mat(invΛ)));
Φmat            = khr2mat(Φ)[:,p][:,1:budget]
p               = sortperm(diag(mpo2mat(invΛs)));
Φsmat           = khr2mat(Φstar)[:,p][:,1:budget]
Λ               = mpo2mat(S)[p,p][1:budget,1:budget].^2
# kernel matrix approximation for given basis function budget
K̃               = Φmat*Λ*Φmat'
K               = covSE(X,X,hyp)
                norm(K-K̃)/norm(K)

tmp             = Φmat'*Φmat + (hyp[3]*mpo2mat(invΛ)[p,p][1:budget,1:budget]);
w_rr            = tmp\(Φmat'*y);
m_rr            = Φsmat * w_rr;
P_rr            = hyp[3]*Φsmat * inv(tmp) * Φsmat';
P_rr            = diag(P_rr);
                norm(ystar-m_rr)/norm(ystar)
SMSE_rr,MSLL_rr = errormeasures(m_rr,P_r,ytest,σ_n)


