## Get SLQ
# git clone  https://github.com/MengLiuPurdue/SLQ
# and
# remove the testing code at the bottom of SLQ.jl and copy it into this directory.
include("SLQ.jl")
include("common.jl")

## Need to run this after intro-spectral.jl
C = cycle_graph(100)
C[1,26] = C[26,1] = 1 # add an edge
C[1,51] = C[51,1] = 1 # add an edge
C[1,76] = C[76,1] = 1 # add an edge
Random.seed!(1)
F = simple_spectral_eigenvectors(C,2;nsteps=20000)
