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

N               = 500;     # number of data points
D               = 1;        # dimensions
Md              = 60;       # number of basis functions per dimension
hyp             = [.5, 1., 0.5];
X,y,f,K         = gensynthdata(N,D,hyp);

## full GP
# normal 
logmarglik      = logmarglik_full_(hyp,X,y)
res             = optimize(logmarglik_full,hyp,iterations=1000)
logmarglik      = logmarglik_full_(Optim.minimizer(res),X,y)

∂1,∂2,∂3        =  ∂full(X,hyp,1)
∂1,∂2,∂3        =  ∂full(X,Optim.minimizer(res),1)

# with exponents
logmarglik      = logmarglik_full_exp_(log.(hyp),X,y)
res             = optimize(logmarglik_full_exp,log.(hyp),iterations=1000)
logmarglik      = logmarglik_full_exp_(Optim.minimizer(res),X,y)

∂1,∂2,∂3        =  ∂full(X,hyp,2)
∂1,∂2,∂3        =  ∂full(X,exp.(Optim.minimizer(res)),2)

# Hilbert-GP
M               = 60;
# normal
logmarglik      = logmarglik_bf_(hyp,X,y,M)
res             = optimize(logmarglik_bf,hyp,iterations=1000)
logmarglik      = logmarglik_bf_(Optim.minimizer(res),X,y,M)

∂1,∂2,∂3        =  ∂bf(X,hyp,M,1)
∂1,∂2,∂3        =  ∂bf(X,Optim.minimizer(res),M,1)

# with exponents
logmarglik      = logmarglik_bf_exp_(log.(hyp),X,y,M)
res             = optimize(logmarglik_bf_exp,hyp,iterations=1000)
logmarglik      = logmarglik_bf_exp_(Optim.minimizer(res),X,y,M)

∂1,∂2,∂3        =  ∂bf(X,hyp,M,2)
∂1,∂2,∂3        =  ∂bf(X,exp.(Optim.minimizer(res)),M,2)

# modified Hilbert-GP
# normal
logmarglik      = logmarglik_bf_mod_(hyp,X,y,M)
res             = optimize(logmarglik_bf,hyp,iterations=1000)
logmarglik      = logmarglik_bf_mod_(Optim.minimizer(res),X,y,M)


