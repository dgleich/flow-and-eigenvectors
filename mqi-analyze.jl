#=
These are pedagogical codes. There are many known inefficiencies and
places to improve things.
=#
##
include("FlowSeed.jl")

## Run some analysis
include("facebook-sbm.jl")
function random_small_network()
  A, groups, xy = fbnet(pop, 2 .^(C+7I), 1000; degreedist=LogNormal(log(3),1))
  filt = xy[:,1] .> -130
  A = A[filt,filt]
  xy = xy[filt,:]
  xy .+= 0.2*randn(size(xy)...)
  A,filt = largest_component(A)
  xy = xy[filt,:]
  A = Float64.(A)
  return A,xy
end
function mqi(A,R)
  volA = sum(A)
  sstart = set_stats(A,R,volA)
  delta = sstart[4]

  d = vec(sum(A,dims=1))
  source = zeros(size(A,1))
  sink = zeros(size(A,1))
  sink = volA*ones(size(A,1))
  sink[R] .= 0 # no edges from R

  println("starting cond=", sstart[4], " = ", sstart[1], "/", sstart[2])

  Sbest = copy(R)
  send = sstart
  while true
    source[R] = delta*d[R]
    S = NonLocalPushRelabel(A,R,source,sink)
    if length(S) > 0
      Sbest = S
      send = set_stats(A,S,volA)
      if delta == send[4] # we probed and got back the same value, so optimal
        break
      end
      delta = send[4]
      println("    step cond=", send[4], " = ", send[1], "/", send[2])
    else
      break
    end
  end
  println("  ending cond=", send[4], " = ", send[1], "/", send[2])
  return Sbest, send
end

function analyze()
  A,xy = random_small_network()
  d = vec(sum(A,dims=1))
  R = xy[:,2] .> 40
  S = mqi(A,findall(R))[1]
  return xy[S,:]
end

xyall = map(x->analyze(), 1:50)
##
include("misc.jl")
using Plots
plot(draw_graph_lines(random_small_network()...)...,linealpha=0.02,framestyle=:none,legend=false,)
for i=1:4
  plot!(draw_graph_lines(random_small_network()...)...,linealpha=0.02,framestyle=:none,legend=false,)
end
#plot(draw_graph_lines(A,xy)...,linealpha=0.02,framestyle=:none,legend=false,)
scatter!(eachcol(vcat(xyall...))..., alpha=0.1, markersize=5, markerstrokewidth=0 )
##
plot!(dpi=300)
savefig("simple-us-study.png")
