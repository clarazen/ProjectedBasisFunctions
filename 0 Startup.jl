# run this code once to add the packages that are used for the experiments

import Pkg; 
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("DelimitedFiles")

# download data from https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/Robot_Identification_Benchmark_Without_Raw_Data.rar
# and save data as "benchmark data/X_kukainv.csv", "benchmark data/y_kukainv.csv", "benchmark data/Xtest_kukainv.csv", "benchmark data/ytest_kukainv.csv"