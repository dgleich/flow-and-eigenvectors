
## Now look at musical artists
using DelimitedFiles
using SparseArrays, MatrixNetworks
include("common.jl")
#names = readlines("artistsim.names")
data = readdlm("artistsim.smat")
xy = readdlm("artistsim.xy")
##
A = sparse(Int.(data[2:end,1]).+1,
           Int.(data[2:end,2]).+1,
           Int.(data[2:end,3]),
           Int(data[1,1]),Int(data[1,1]))
A = max.(A,A')
A,subset = largest_component(A)
xy = xy[subset,:]
## Code from ACL from another old code
using Random, LinearAlgebra, DataStructures
function simple_acl(A::SparseMatrixCSC{T,Int}, R,
  alpha::Float64, eps::Float64; dvec::Vector{Int}=vec(sum(A;dims=2))) where T

  n = size(A,1)
  x = zeros(n) # TODO, can use dicts for sparsity
  r = zeros(n)
  Q = Queue{Int}()
  Rvol = 0
  for s in R
    enqueue!(Q,s)
    Rvol += dvec[s]
  end
  for s in R
    r[s] = dvec[s] / Rvol
  end

  colptr,rowval = A.colptr, A.rowval
  @inbounds while length(Q) > 0
    u = dequeue!(Q)
    du = dvec[u] # get the degree
    pushval = r[u] - 0.5*eps*du
    x[u] +=  (1-alpha)*pushval
    r[u] = 0.5*eps*du
    pushval = pushval*alpha/du
    for nzi in colptr[u]:(colptr[u+1] - 1)
      v = rowval[nzi]
      dv = dvec[v] # degree of v
      rvold = r[v]
      rvnew = rvold + pushval
      r[v] = rvnew
      if rvnew > eps*dv && rvold <= eps*dv
        enqueue!(Q,v)
      end
    end
  end
  return x
end
function myscatter!(xy,z;kwargs...)
  p = sortperm(z)
  z = z[p]
  xy = xy[p,:]
  f = z .> 0
  scatter!(xy[f,1], xy[f,2], marker_z = log10.(z[f]); kwargs...)
end
myscatter!(xy, x, markersize=6, markeralpha=0.1, markerstrokewidth=0)
##
x=simple_acl(A,[8],0.99,1e-5)
plot(size=(1200,1200))
myscatter!(xy, x, markerstrokewidth=0)
##
x=simple_acl(A,[8],0.99,1e-5)
plot(graph_lines(A,xy)...,size=(1200,1200),
  linecolor=1, linealpha=0.01, linewidth=0.5,
  framestyle=:none,legend=false)
myscatter!(xy, x, markersize=6, markeralpha=0.4, markerstrokewidth=0)
##
using Printf
anim = @animate for pr_eps = 10.0.^(range(-0.5,-5,length=60))
  x = simple_acl(C,[8],0.99,pr_eps)
  plot(graph_lines(A,xy)...,size=(1200,1200),
    linecolor=1, linealpha=0.01, linewidth=0.5,
    framestyle=:none,legend=false)
  myscatter!(xy, x, markersize=6, markeralpha=0.4, markerstrokewidth=0)
  title!(@sprintf("eps = %.2e", pr_eps))
end
gif(anim, "acl-artistsim-eps.gif")

## Illustrating ACL on cycle graph
C = cycle_graph(100)
C[1,51] = C[51,1] = 1 # add an edge
Random.seed!(1)
F = simple_spectral_eigenvectors(C,2;nsteps=20000)
##
x = simple_acl(C,R,0.99,1e-3)
plot(graph_lines(C,F)...,
  framestyle=:none,
  markersize=5,marker=:dot,legend=false)
myscatter!(F,x,markersize=8)
##
using Printf
anim = @animate for pr_eps = 10.0.^(range(-0.5,-4,length=60))
  x = simple_acl(C,R,0.99,pr_eps)
  plot(graph_lines(C,F)...,
    framestyle=:none,
    markersize=5,marker=:dot,legend=false)
  myscatter!(F,x,markersize=8)
  title!(@sprintf("eps = %.2e", pr_eps))
end
gif(anim, "acl-cycle-eps.gif")

## Animate the ACL algorithm on a clover graph
C = cycle_graph(100)
C[1,26] = C[26,1] = 1 # add an edge
C[1,51] = C[51,1] = 1 # add an edge
C[1,76] = C[76,1] = 1 # add an edge
Random.seed!(1)
F = simple_spectral_eigenvectors(C,2;nsteps=20000)
##
R = [14]
x = simple_acl(C,R,0.99,1e-2)

function animated_simple_acl(A::SparseMatrixCSC{T,Int}, R, plotfun,
  alpha::Float64, eps::Float64; dvec::Vector{Int}=vec(sum(A;dims=2)),
  every::Int=1) where T

  n = size(A,1)
  x = zeros(n) # TODO, can use dicts for sparsity
  r = zeros(n)
  Q = Queue{Int}()
  Rvol = 0
  for s in R
    enqueue!(Q,s)
    Rvol += dvec[s]
  end
  for s in R
    r[s] = dvec[s] / Rvol
  end
  npush = 0
  anim = Animation()

  plotfun(x,r,npush)
  frame(anim)


  colptr,rowval = A.colptr, A.rowval
  @inbounds while length(Q) > 0
    u = dequeue!(Q)
    du = dvec[u] # get the degree
    pushval = r[u] - 0.5*eps*du
    x[u] +=  (1-alpha)*pushval
    r[u] = 0.5*eps*du
    pushval = pushval*alpha/du
    for nzi in colptr[u]:(colptr[u+1] - 1)
      v = rowval[nzi]
      dv = dvec[v] # degree of v
      rvold = r[v]
      rvnew = rvold + pushval
      r[v] = rvnew
      if rvnew > eps*dv && rvold <= eps*dv
        enqueue!(Q,v)
      end
    end
    npush += 1
    if mod(npush, every) == 0
      plotfun(x,r,npush)
      frame(anim)
    end
  end
  return x, anim
end
function aclplotfuncycle(x,r,iter)
  plot(graph_lines(C,F)...,
    framestyle=:none,
    markersize=0,marker=:dot,legend=false)
  dvec = vec(sum(C;dims=2))
  rrel = r./(pr_eps*dvec)
  scatter!(F[:,1],F[:,2],
    markersize = 3 .+ 3*(r .< pr_eps*dvec).*rrel .+ 8*(r .> pr_eps*dvec).*(1.0.+log10.(rrel)),
    color=1)
  title!("npush = $iter")
  #scatter!(xy[f,1], xy[f,2], marker_z = log10.(z[f]); kwargs...)
  myscatter!(F,x,markersize=8)
end

R = [7]
pr_eps = 1e-2
x,anim = animated_simple_acl(C,R,aclplotfuncycle,0.99,pr_eps)
gif(anim, "acl-alg-clover.gif", fps=15)

##
using Printf
anim = @animate for pr_eps = 10.0.^(range(-0.5,-4,length=60))
  x = simple_acl(C,R,0.99,pr_eps)
  plot(graph_lines(C,F)...,
    framestyle=:none,
    markersize=5,marker=:dot,legend=false)
  myscatter!(F,x,markersize=8)
  title!(@sprintf("eps = %.2e", pr_eps))
end
gif(anim, "acl-clover-eps.gif")
