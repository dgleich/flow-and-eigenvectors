## Get code for spectral clustering from Purdue CS590-NCDS
# https://github.com/dgleich/cs590-ncds
# https://github.com/dgleich/cs590-ncds/blob/master/5-unit-4-demos/spectral-clustering-simple.jl
# update to get multiple eigenvectors

## Get code for spectral clustering from Purdue CS590-NCDS
# https://github.com/dgleich/cs590-ncds
# https://github.com/dgleich/cs590-ncds/blob/master/5-unit-4-demos/spectral-clustering-simple.jl
using Random, LinearAlgebra
Random.seed!(1)

##
function cycle_graph(n::Int)
  A = sparse(1:n-1,2:n,1,n,n)
  A[1,end] = 1
  A = max.(A,A')
  return A
end
##
using Plots
include("common.jl")
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

## Make a movie of the convergence
function simple_spectral_eigenvectors_movie(A,k,plotfun;nsteps=500,every::Int=1)
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
  X .*= repeat(sign.(X[1,:])',size(X,1),1)
  anim = Animation()
  plotfun(X)
  frame(anim)
  for i=1:nsteps
    Xold = copy(X)
    X .+= (A*X)./d     # matrix-vector with (I + AD^{-1}) X
    X .-= ((d'*X)/nd2).*d
    Q = qr!(sqrt.(d).*X).Q
    X = sqrt.(1.0./d).*Matrix(Q)
    # figure out signs
    bestdiff = norm(X .- Xold)
    bestsign = [1,1]
    for signs = [[-1,-1],[-1,1],[1,-1]]
      if norm(repeat(signs',size(X,1),1).*X - Xold) < bestdiff
        bestsign = signs
      end
    end
    X = repeat(bestsign',size(X,1),1).*X
    if mod(i,every)==0
      plotfun(X)
      frame(anim)
    end
  end
  # make sure the first component is positive
  X .*= repeat(sign.(X[1,:])',size(X,1),1)

  return X, anim
end
plotfun = F->plot(graph_lines(C,F)...,
  markersize=5,marker=:dot,
  framestyle=:none,legend=false)
F,anim = simple_spectral_eigenvectors_movie(C,2,plotfun;nsteps=60)
gif(anim, "intro-spectral-cycle-converge.gif")
##
C = cycle_graph(100)
C[1,51] = C[51,1] = 1 # add an edge
plotfun = F->plot(graph_lines(C,F)...,
  markersize=5,marker=:dot,
  framestyle=:none,legend=false)
Random.seed!(1)
F,anim = simple_spectral_eigenvectors_movie(C,2,plotfun;nsteps=300)
gif(anim, "intro-spectral-long-cycle-converge.gif")
##
Random.seed!(1)
F,anim = simple_spectral_eigenvectors_movie(C,2,plotfun;nsteps=60)
gif(anim, "intro-spectral-long-cycle-converge-60.gif")
##
F = simple_spectral_eigenvectors(C,2;nsteps=20000)
plot(graph_lines(C,F)...,
  markersize=5,marker=:dot,
  framestyle=:none,legend=false)
plot!(size=(400,400))
savefig("intro-spectral-long-cycle-100.pdf")
##
C = cycle_graph(100)
C[1,51] = C[51,1] = 1 # add an edge
plotfun = F->plot(graph_lines(C,F)...,
  markersize=5,marker=:dot,
  framestyle=:none,legend=false)
Random.seed!(1)
F,anim = simple_spectral_eigenvectors_movie(C,2,plotfun;nsteps=300)
gif(anim, "intro-spectral-long-cycle-converge.gif")
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
plotfun = F->plot(graph_lines(A,F)...,marker=:dot,
  markersize=3,markerstrokewidth=0,linewidth=0.5,linealpha=0.1,markeralpha=0.3,
  framestyle=:none,legend=false)
F,anim = simple_spectral_eigenvectors_movie(A,2,plotfun;nsteps=100)
gif(anim, "intro-spectral-artistsim-converge.gif")
## show a picture

##
plot!(dpi=300,size=(400,400))
savefig("intro-spectral-artistsim.png")

## 6:30am
