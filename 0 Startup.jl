# run this code once to add the packages that are used for the experiments
using Pkg
Pkg.add(url="https://github.com/clarazen/BigMat.git") # this does not add what is online but an older version for some reason
Pkg.add(url="https://github.com/clarazen/TN4GP.git")

# also install all packages that you have not installed yet.
import Pkg; 
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("Optim")
Pkg.add("DelimitedFiles")
Pkg.add("StatsBase")
Pkg.add("Distributions")
Pkg.add("GaussianProcesses")



