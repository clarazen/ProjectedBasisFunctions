using LinearAlgebra
using Revise
using Pkg
Pkg.activate("C:/Users/cmmenzen/.julia/dev/BigMat")
using BigMat
Pkg.activate("C:/Users/cmmenzen/.julia/dev/TN4GP/")
using TN4GP
using Plots
using Optim
using DelimitedFiles
using StatsBase
using Distributions


R           = 20;
N           = 5000;     # number of data points
D           = 3;       # dimensions
M           = 40;       # number of basis functions per dimension
hyp         = [0.05*ones(D), 1., 0.01];
Xall,~,~,~  = gensynthdata(N,D,hyp);

boundsMin   = minimum(Xall,dims=1);
boundsMax   = maximum(Xall,dims=1);
L           = 2*((boundsMax.-boundsMin) ./ 2)[1,:] #.+ ((boundsMax.-boundsMin) ./ 4)[1,:];

Φ_          = colectofbasisfunc(M*ones(D),Xall,hyp[1],hyp[2],L,true);
Φ_mat       = khr2mat(Φ_);

K̃           = Φ_mat*Φ_mat';
K           = covSE(Xall,Xall,hyp);
            norm(K-K̃)/norm(K)

#w,err       = MPT_SVD(mps2vec(TT_ALS(randn(M^3),[M M M],[1,R,R,1])),[M M M],1e-15);
#            shiftMPTnorm(w,3,-1)
#Wd          = mpo2mat(getU(w,2))
#fall        = Φ_mat*Wd*randn(length(w[2]));

w1          = Matrix(qr(randn(M,R)).Q);
w2          = randn(R*M*R);
w3          = Matrix(qr(randn(M,R)).Q);
Wd          = kron(kron(w3,Matrix(I,M,M)),w1)

Φ_matWd = Φ_mat*Wd
norm(Φ_matWd*Φ_matWd'-K)/norm(K)

fall        = Φ_mat*w2
yall        = fall + sqrt(hyp[3])*randn(size(fall))

X           = Xall[1:4000,:];
Xstar       = Xall[4001:5000,:];
y           = yall[1:4000];
ystar       = fall[4001:5000];

Φ_          = colectofbasisfunc(M*ones(D),X,hyp[1],hyp[2],L,true);
Φ_mat       = khr2mat(Φ_);
Φstar_      = colectofbasisfunc(M*ones(D),Xstar,hyp[1],hyp[2],L,true);
Φstar_mat   = khr2mat(Φstar_);

# full GP
K           = covSE(X,X,hyp);
mstar,Pstar = fullGP(K,X,Xstar,y,hyp,false);
#p = sortperm(mstar-ystar)
#scatter(abs.(mstar-ystar)[p],Pstar[p])
#scatter((mstar-ystar) ./ Pstar)
            norm(mstar-ystar)/norm(ystar)

# tt approximation
# ALS to find weight matrix
rnks        = Int.([1, R*ones(D-1,1)..., 1]);
maxiter     = 10;
@time tt,res  =  ALS_krtt_mod(y,Φ_,rnks,maxiter,hyp[3],hyp[3]);
m_tt        = Φstar_mat*mps2vec(tt);
            norm(y - Φ_mat*mps2vec(tt))/norm(y)
            norm(ystar-m_tt)/norm(ystar)
# Bayesian update of second core
            shiftMPTnorm(tt,D,-1);
ttm         = getU(tt,2);   # works       
U           = krtimesttm(Φ_,transpose(ttm)); # works
W2          = mpo2mat(ttm);
tmp         = U*U';
tmp         = tmp + hyp[3]*Matrix(I,size(tmp));     
tt[2]       = reshape(tmp\(U*y),size(tt[2]))
m_tt        = Φstar_mat*mps2vec(tt);
cova2       = inv(tmp);
Φstarttm    = krtimesttm(Φstar_,transpose(ttm));
P_tt        = hyp[3]*Φstarttm'*cova2*Φstarttm;
P_tt        = 2*sqrt.(diag(P_tt));
cov_tt      = (ystar-m_tt) ./ Pstar
err_tt      = norm(ystar-m_tt)/norm(ystar)

## rr approximation
budget          = R*R*M;
budgetd         = Int.(ceil((budget)^(1/D)))
Φ,invΛ,S        = colectofbasisfunc(budgetd*ones(D),X,hyp[1],hyp[2],L);
Φstar,invΛs,~   = colectofbasisfunc(budgetd*ones(D),Xstar,hyp[1],hyp[2],L);
p               = sortperm(diag(mpo2mat(invΛ)));
Φmat            = khr2mat(Φ)[:,p][:,1:budget]
p               = sortperm(diag(mpo2mat(invΛs)));
Φsmat           = khr2mat(Φstar)[:,p][:,1:budget]
Λ               = mpo2mat(S)[p,p][1:budget,1:budget].^2

K̃               = Φmat*Λ*Φmat'
K               = covSE(X,X,hyp)
                norm(K-K̃)/norm(K)

                # sth is not right here maybe
tmp             = Φmat'*Φmat + (hyp[3]*mpo2mat(invΛ)[p,p][1:budget,1:budget]);
w_rr            = tmp\(Φmat'*y);
m_rr            = Φsmat * w_rr;
P_rr            = hyp[3]*Φsmat * inv(tmp) * Φsmat';
P_rr            = 2*sqrt.(diag(P_rr));
err_rr          = norm(ystar-m_rr)/norm(ystar)
cov_rr          = (ystar-m_rr) ./ Pstar



