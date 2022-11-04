function errormeasures(mstar::Vector,Pstar::Vector,ytest::Vector,σ_n::Float64)

    # !! Pstar needs to be the variance of the prediction
    SMSE = 0
    MSLL = 0
    for i = 1:length(mstar)
        SMSE = SMSE + ( (ytest[i] - mstar[i])^2 / var(ytest) )
        MSLL = MSLL + ( (ytest[i] - mstar[i])^2 / (Pstar[i] + σ_n^2) + log(2*π*(Pstar + σ_n^2)) / 2length(mstar) )
    end

    return SMSE,MSLL
end

