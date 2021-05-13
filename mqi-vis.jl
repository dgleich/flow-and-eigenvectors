#=
These are pedagogical codes. There are many known inefficiencies and
places to improve things.
=#
##
include("FlowSeed.jl")
##
function MQITest(A,R,delta)
  volA = sum(A)
  d = vec(sum(A,dims=1))

  source = zeros(size(A,1))
  source[R] = delta*d[R] # build edge weights from source to R

  sink = volA*ones(size(A,1)) # build edge weights from Rbar to sink
  sink[R] .= 0 # no edges from R
  sstart = set_stats(A,R,volA) # get conductance
  println("starting cond=", sstart[4], " = ", sstart[1], "/", sstart[2])
  S = NonLocalPushRelabel(A,R,source,sink)
  if length(S) > 0
    send = set_stats(A,S,volA)
    println("  ending cond=", send[4], " = ", send[1], "/", send[2])
  else
    send = sstart
    S = R
  end
  return S, send
end

## Create a graph we can visualize via a facebook network
include("facebook-sbm.jl")
using Random
Random.seed!(1)
A, groups, xy = fbnet(pop, 2 .^(C+7I), 1000; degreedist=LogNormal(log(3),1))
filt = xy[:,1] .> -140
A = A[filt,filt]
xy = xy[filt,:]
xy .+= 0.2*randn(size(xy)...)
A,filt = largest_component(A)
xy = xy[filt,:]
A = Float64.(A)

##
include("misc.jl")
using Plots
plot(draw_graph_lines(A,xy)...,linealpha=0.1,framestyle=:none,legend=false,)
scatter!(xy[:,1],xy[:,2],markersize=2, markerstrokewidth=0, alpha=0.25,color=7)

##
plot!(dpi=300)
savefig("simple-us.png")

##
A = Float64.(A) # convert to float
##
d = vec(sum(A,dims=1))
R = xy[:,2] .> 40
@show sum(R), sum(d.*R), sum(d)
##
plot(draw_graph_lines(A,xy)...,linealpha=0.1,framestyle=:none,legend=false,)
scatter!(xy[:,1],xy[:,2],markersize=2, markerstrokewidth=0, alpha=0.25,color=7)
scatter!(xy[R,1],xy[R,2],markersize=8, markerstrokewidth=0, alpha=0.25,color=2)
##
plot!(dpi=300)
savefig("simple-us-R.png")
##
S1, s1stat = MQITest(A,findall(R),0.208)
##
function mqi_set(A,xy,R,S)
  #plot(draw_graph_lines(A,xy)...,linealpha=0.1,framestyle=:none,legend=false,)
  #Ain = copy(A)
  #Arest = A - Ain - Aboundary
  #A[]
  Arest = copy(A)
  Arest[S,:] .= 0
  Arest[:,S] .= 0
  ASwithBoundary = A - Arest
  ABoundary = copy(ASwithBoundary)
  ABoundary[S,S] .= 0
  dropzeros!(ABoundary)
  AS = ASwithBoundary-ABoundary
  dropzeros!(AS)
  plot(draw_graph_lines(Arest,xy)...,linealpha=0.1,framestyle=:none,legend=false)
  plot!(draw_graph_lines(AS,xy)...,linealpha=0.2,linewidth=1.5,framestyle=:none,legend=false,color=:black)
  plot!(draw_graph_lines(ABoundary,xy)...,linealpha=0.2,linewidth=2,framestyle=:none,legend=false,color=2)

  scatter!(xy[:,1],xy[:,2],markersize=2, markerstrokewidth=0, alpha=0.25,color=7)
  scatter!(xy[R,1],xy[R,2],markersize=4, markerstrokewidth=0, alpha=0.25,color=2)
  scatter!(xy[S,1],xy[S,2],markersize=8, markerstrokewidth=0, alpha=0.25,color=3)
end
mqi_set(A,xy,R,S1)
##
plot!(dpi=300)
savefig("simple-us-S1.png")

##
S2, s2stat = MQITest(A,findall(R),0.152)
##
mqi_set(A,xy,R,S2)
plot!(dpi=300)
savefig("simple-us-S2.png")
##
S3, s3stat = MQITest(A,findall(R),0.13)
##
mqi_set(A,xy,R,S3)
plot!(dpi=300)
savefig("simple-us-S3.png")
##
S4, s4stat = MQITest(A,findall(R),0.124)
##
mqi_set(A,xy,R,S4)
plot!(dpi=300)
savefig("simple-us-S4.png")
##
S5, s5stat = MQITest(A,findall(R),0.120)
##
mqi_set(A,xy,R,S5)
plot!(dpi=300)
savefig("simple-us-S5.png")
##
S6, s6stat = MQITest(A,findall(R),0.115)
##
mqi_set(A,xy,R,S6)
plot!(dpi=300)
savefig("simple-us-S6.png")

##
Sf,sstats = MQITest(A,findall(R),0.111)
##
plot(draw_graph_lines(A,xy)...,linealpha=0.1,framestyle=:none,legend=false,)
scatter!(xy[:,1],xy[:,2],markersize=2, markerstrokewidth=0, alpha=0.25,color=7)
scatter!(xy[Sf,1],xy[Sf,2],markersize=8, markerstrokewidth=0, alpha=0.25,color=2)

## Run some analysis
function random_small_network()
  A, groups, xy = fbnet(pop, 2 .^(C+7I), 1000; degreedist=LogNormal(log(3),1))
  filt = xy[:,1] .> -140
  A = A[filt,filt]
  xy = xy[filt,:]
  xy .+= 0.2*randn(size(xy)...)
  A,filt = largest_component(A)
  xy = xy[filt,:]
  return A,xy
end
function mqi(A,R)
  volA = sum(A)
  sstart = set_stats(A,R,volA)
  delta = sstart[4]

  d = vec(sum(A,dims=1))
  source = zeros(size(A,1))
  source[R] = delta*d[R]
  sink = zeros(size(A,1))
  sink = volA*ones(size(A,1))
  sink[R] .= 0 # no edges from R

  println("starting cond=", sstart[4], " = ", sstart[1], "/", sstart[2])

  Sbest = copy(R)
  while true
    S = NonLocalPushRelabel(A,R,source,sink)
    if length(S) > 0
      Sbest = S
      send = set_stats(A,S,volA)
      println("    step cond=", send[4], " = ", send[1], "/", send[2])
    else
      println("  ending cond=", send[4], " = ", send[1], "/", send[2])
      break
    end
  end
  return S, send
end
