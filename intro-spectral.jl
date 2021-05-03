## Get code for spectral clustering from Purdue CS590-NCDS
# https://github.com/dgleich/cs590-ncds
# https://github.com/dgleich/cs590-ncds/blob/master/5-unit-4-demos/spectral-clustering-simple.jl
# update to get multiple eigenvectors

## Get code for spectral clustering from Purdue CS590-NCDS
# https://github.com/dgleich/cs590-ncds
# https://github.com/dgleich/cs590-ncds/blob/master/5-unit-4-demos/spectral-clustering-simple.jl
using Random, LinearAlgebra
Random.seed!(1)
function simple_spectral_eigenvectors(A,k;nsteps=500,tol=1e-6)
  @assert issymmetric(A) # expensive, but useful...
  n = size(A,1)
  d = vec(sum(A,dims=1))
  nd2 = d'*d
  X = randn(n,k)
  # project x orthogonal to d
  X .-= ((d'*X)/nd2).*d
  #x ./= x'*(d.*x) # normalize
  Q = qr!(sqrt.(d).*X).Q
  X = sqrt.(1.0./d).*Matrix(Q)
  for i=1:nsteps
    X .+= (A*X)./d     # matrix-vector with (I + AD^{-1}) X
    X .-= ((d'*X)/nd2).*d
    Q = qr!(sqrt.(d).*X).Q
    X = sqrt.(1.0./d).*Matrix(Q)
  end
  # make sure the first component is positive
  X .*= repeat(sign.(X[1,:])',size(X,1),1)

  return X
end
##
function cycle_graph(n::Int)
  A = sparse(1:n-1,2:n,1,n,n)
  A[1,end] = 1
  A = max.(A,A')
  return A
end
##
C = cycle_graph(10)
C[1,6] = C[6,1] = 1 # add an edge
F = simple_spectral_eigenvectors(C,2)
plot(graph_lines(C,F)...,
  markersize=5,marker=:dot,
  framestyle=:none,legend=false)
##
plot!(size=(400,400))
savefig("intro-spectral-cycle.pdf")
## Woohoo, code works, spectral clustering rocks!

## Now look at musical artists
using DelimitedFiles
using SparseArrays
#names = readlines("artistsim.names")
data = readdlm("artistsim.smat")
##
A = sparse(Int.(data[2:end,1]).+1,
           Int.(data[2:end,2]).+1,
           Int.(data[2:end,3]),
           Int(data[1,1]),Int(data[1,1]))
A = max.(A,A')
A = largest_component(A)[1]
##
@time F = simple_spectral_eigenvectors(A,2;nsteps=5000)
## show a picture
include("common.jl")
plot(graph_lines(A,F)...,marker=:dot,
  markersize=3,markerstrokewidth=0,linewidth=0.5,linealpha=0.1,markeralpha=0.3,
  framestyle=:none,legend=false)
##
plot!(dpi=300,size=(400,400))
savefig("intro-spectral-artistsim.png")

##
