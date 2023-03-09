module functionsMetrics

using StatsBase

export MSLL,SMSE

function MSLL(mstar::Vector,Pstar::Vector,ytest::Vector,σ_n::Float64)

    # !! Pstar needs to be the variance of the prediction
    MSLL = 0
    for i = 1:length(mstar)
        MSLL = MSLL + ( (ytest[i] - mstar[i])^2 / (Pstar[i] + σ_n^2) + log(2*π*(Pstar[i] + σ_n^2)) / 2length(mstar) )
    end

    return MSLL
end

function SMSE(mstar::Vector,ytest::Vector,y::Vector)

    SMSE = 0
    for i = 1:length(mstar)
        SMSE = SMSE + ( (ytest[i] - mstar[i])^2 / var(y) )
    end

    return SMSE
end

function RMSE()
end

end

function RMSE(mstar::Vector,ytest::Vector)
    SMSE = 0
    for i = 1:length(mstar)
        SMSE = SMSE + (ytest[i] - mstar[i])^2
    end
    return sqrt(SMSE/length(mstar))
end