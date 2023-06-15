
Xtrain     = readdlm("C:/Users/cmmenzen/.julia/dev/UCIdatasets/energy_Xtrain_0.csv",',')
ytrain     = readdlm("C:/Users/cmmenzen/.julia/dev/UCIdatasets/energy_ytrain_0.csv",',')
Xtest      = readdlm("C:/Users/cmmenzen/.julia/dev/UCIdatasets/energy_Xtest_0.csv",',')
ytest      = readdlm("C:/Users/cmmenzen/.julia/dev/UCIdatasets/energy_ytest_0.csv",',')

function standardscaler(Xtrain,Xtest,ytrain,ytest)

    # normalize y scale and offset
    ymean       = mean(ytrain)
    ystd        = std(ytrain)
    ytrain_st   = (ytrain .- ymean) ./ ystd;
    ytest_st    = (ytest  .- ymean) ./ ystd;
    # normalize X on [-0.5 0.5]
    xmin        = minimum(Xtrain,dims=1);
    xmax        = maximum(Xtrain,dims=1);
    Xtrain_st   = (Xtrain .- xmin) ./ (xmax - xmin) .- .5;
    Xtest_st    = (Xtest .- xmin)  ./ (xmax - xmin) .- .5;

    return Xtrain_st,Xtest_st,ytrain_st,ytest_st
end

