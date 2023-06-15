using DelimitedFiles
using StatsBase
using XLSX

data     = readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/kin40k.csv",',')
data     = readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/elevator.csv",',')
data     = readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/houseelectric.csv",',')
data     = readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/housing.csv",',')
data     = readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/keggdirected.csv",',')
data     = readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/izmailov/protein.csv",',')[2:end,:]
data     = readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/izmailov/YearPredictionMSD.txt",',')
xf       = XLSX.readxlsx("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/izmailov/powerplant/Folds5x2_pp.xlsx");
data     = Float64.(xf["Sheet1"]["A2:E9569"]);
data     = readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/airline.csv",',')
data     = readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/concrete.csv",',')
data     = readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/yacht.csv",',')[1:end-1,:]
data     = readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/airfoil.csv",',')
data     = Float64.(readdlm("C:/Users/cmmenzen/surfdrive/Code/UCI data sets/winequality-red.csv",';')[2:end,:])

# test data
#data = [1 2 3 10; 4 5 6 11; 7 8 9 12; 0.1 0.2 0.3 1; .4 .5 .6 1.1; .7 .8 .9 1.2; 0.01 0.02 0.03 1.01; .04 .05 .06 1.11]

Xtotn,ytotn                             = standardscaler(data);
X,Xtest,y,ytest, Xsub, Xval, ysub, yval = foldsfornestedXV(Xtotn,ytotn,5);

function standardscaler(data::Matrix)
    # exctract X and y
    Xtot     = data[:,1:end-1]; # for powerplant, house, kin40k (i guess), concrete
    ytot     = data[:,end]; # for powerplant, house, concrete
    #Xtot     = data[:,2:end]; # for protein, YearPredictionMSD
    #ytot     = data[:,1]; # for protein, YearPredictionMSD
    Ntot     = size(ytot,1);
    # normalize y scale and offset
    ymean    = mean(ytot)
    ystd     = std(ytot)
    ytotn    = (ytot .- ymean) ./ ystd;
    # normalize X on [-0.5 0.5]
    xmin    = minimum(Xtot,dims=1);
    xmax    = maximum(Xtot,dims=1);
    Xtotn   = (Xtot .- xmin) ./ (xmax - xmin) .- .5;

    return Xtotn,ytotn
end

function foldsfornestedXV(Xtotn::Matrix,ytotn::Vector,folds::Int)
    # nested cross validation for hyper parameter tuining and model evaluation
    N        = Int(ceil(Ntot / folds));
    X_folds  = Vector{Matrix}(undef,folds);
    y_folds  = Vector{Vector}(undef,folds);
    for i = 1:folds-1 # create different data chunks
        X_folds[i] = Xtotn[(i-1)*N+1 : i*N,:]
        y_folds[i] = ytotn[(i-1)*N+1 : i*N]
    end
    X_folds[folds]    = Xtotn[(folds-1)*N+1 : end,:]
    y_folds[folds]    = ytotn[(folds-1)*N+1 : end]

    # creating folds for outer loop
    X               = Matrix{Matrix}(undef,folds,folds-1);
    y               = Matrix{Vector}(undef,folds,folds-1);
    Xtest           = Vector{Matrix}(undef,folds);
    ytest           = Vector{Vector}(undef,folds);

    X_folds         = [X_folds;X_folds];
    y_folds         = [y_folds;y_folds];
    for i = 1:folds
        for j = 1:folds-1
            X[i,j]  = X_folds[i+j-1]
            y[i,j]  = y_folds[i+j-1]
        end
        Xtest[i]    = X_folds[i+folds-1]
        ytest[i]    = y_folds[i+folds-1]
    end

    # inner loop
    Xsub           = Vector{Matrix{Matrix}}(undef,folds);
    ysub           = Vector{Matrix{Vector}}(undef,folds);
    Xval           = Vector{Vector{Matrix}}(undef,folds);
    yval           = Vector{Vector{Vector}}(undef,folds);

    for i = 1:folds
        Xouter         = [X[i,:]; X[i,:]];
        youter         = [y[i,:]; y[i,:]];
        Xsub_i         = Matrix{Matrix}(undef,folds-1,folds-2)
        ysub_i         = Matrix{Vector}(undef,folds-1,folds-2)
        Xval_i         = Vector{Matrix}(undef,folds-1)
        yval_i         = Vector{Vector}(undef,folds-1)

        for j = 1:folds-1
            for k = 1:folds-2
                Xsub_i[j,k]  = Xouter[j+k-1]
                ysub_i[j,k]  = youter[j+k-1]
            end
            Xval_i[j]    = Xouter[j+folds-2]
            yval_i[j]    = youter[j+folds-2]
        end
        Xsub[i]    = Xsub_i;
        ysub[i]    = ysub_i;
        Xval[i]    = Xval_i;
        yval[i]    = yval_i;
    end
    return X,Xtest,y,ytest, Xsub, Xval, ysub, yval
end