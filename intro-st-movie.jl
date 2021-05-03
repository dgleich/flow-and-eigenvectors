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

## Make a movie of the convergence
function simple_st_eigenvectors_movie(A,R,k,plotfun;nsteps=500,every::Int=1)
  @assert issymmetric(A) # expensive, but useful...
  n = size(A,1)
  d = vec(sum(A,dims=1))
  nd2 = d'*d
  X = zeros(n,k)
  X[R,1] .= 1
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
C = cycle_graph(100)
C[1,51] = C[51,1] = 1 # add an edge
R = collect(4:6)
plotfun = F->begin
  plot(graph_lines(C,F)...,
    framestyle=:none,
    markersize=5,marker=:dot,legend=false)
  scatter!(F[R,1], F[R,2],markersize=8)
end
F,anim = simple_st_eigenvectors_movie(C,R,2,plotfun;nsteps=60)
gif(anim, "intro-st-cycle-converge.gif")

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
