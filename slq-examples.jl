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
##
using Printf
anim = @animate for slq_eps = 10.0.^(range(-0.5,-4,length=60))
  x = SLQ.slq_diffusion(SLQ.graph(C), R, 0.1, slq_eps, 0.5,
        SLQ.loss_type(1.02,0.0))[1]
  coords = F
  plot(graph_lines(C,coords)...,
    framestyle=:none,
    markersize=5,marker=:dot,legend=false)
  myscatter!(coords,x,markersize=8)
  title!(@sprintf("eps = %.2e", slq_eps))
end
gif(anim, "slq-clover-eps.gif")
##
using Printf
anim = @animate for slq_eps = 10.0.^(range(-0.5,-4,length=60))
  x = SLQ.slq_diffusion(SLQ.graph(C), R, 0.1, slq_eps, 0.5,
        SLQ.loss_type(1.02,0.0))[1]
  coords = [log10.(x) F[:,2]]
  plot(graph_lines(C,coords)...,
    framestyle=:none,
    markersize=5,marker=:dot,legend=false)
  myscatter!(coords,x,markersize=8)
  title!(@sprintf("eps = %.2e", slq_eps))
end
gif(anim, "slq-clover-eps-shift.gif")
##
using Printf
anim = @animate for pr_eps = 10.0.^(range(-0.5,-4,length=60))
  x = simple_acl(C,R,0.99,pr_eps)
  coords = [log10.(x) F[:,2]]
  plot(graph_lines(C,coords)...,
    framestyle=:none,
    markersize=5,marker=:dot,legend=false)
  myscatter!(coords,x,markersize=8)
  title!(@sprintf("eps = %.2e", pr_eps))
end
gif(anim, "acl-clover-eps-shift.gif")
