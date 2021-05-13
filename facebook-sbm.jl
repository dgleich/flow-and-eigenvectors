#=
Goal, create an SBM from the Facebook county data.
=#

# You may have to update these to the 'more-graphs' path on your
# own computer!
# See
# https://github.com/dgleich/more-graphs

using MatrixNetworks, SparseArrays, LinearAlgebra, DelimitedFiles
fbdata = "$(homedir())/Dropbox/data/more-graphs/facebook-counties/"
##
C = MatrixNetworks.readSMAT("$(fbdata)/facebook-county-friendship.smat")
meta = readdlm("$(fbdata)/facebook-county-friendship-metadata.csv")
xyC = readdlm("$(fbdata)/facebook-county-friendship.xy")
## Interpret the values of C as probabilities of an SBM.
pop = Float64.(meta[:,1])
## Generete a sample of countys by
using StatsBase, Distributions

"""
This is a slightly funky way to generate an SBM sample. See more in
facebook-sbm-1.jl

This is not the best / most efficient way. It also isn't quite correct.
But it's a good match where you don't have exactly an SBM to start with.

We sample a group for each vertex. Then sort them so groups
  are adjacent. This is only to optimize the next step where we build
  sampling weights for each group.

Then for each group, we build a sampling distribution for the other vertices.
  This takes O(n) where there are n vertices. (Or maybe n-log-n) so we get
  K*n log(n).

Then we sample a degree for each node from degree-dist. And then get
  x samples from the sampling distribution. There could be duplicates, etc.
  Do with these as you see fit (maybe easiest to ignore them.)

This function builds a sampling distribution for each group.
  The runtime is O(K^2 + E log(n)) where there are K groups in the SBM
  and E edges output assuming we can sample from groupw in log(n) time,
  which is definitely possible...

Parameters:
pop - a set of parameters that can be converted into FrequencyWeights for
sampling from the groups. This could be handled a different way.
C - the matrix of group liklihoods; these need not be probabilities and could
be numeric scores too.
n - the number of vertices to sample.
"""
function genedges(pop, C, n; degreedist = LogNormal(log(25/2), 2))
  w = FrequencyWeights(pop)
  groups = sort(sample(1:length(pop), w, n))

  src = Vector{Int}()
  dst = copy(src)

  curgroup = 0
  groupw = FrequencyWeights(C[:,1]) # this samples neighbors.
  for i=1:n
    gi = groups[i]
    if gi != curgroup
      # rebuild groupw
      groupw = FrequencyWeights([C[g,gi] for g in groups])
      curgroup = gi
    end
    nneighs = ceil(Int,rand(degreedist))
    neighs = sample(1:n, groupw, nneighs)
    for j in neighs
      if i != j
        push!(src, i)
        push!(dst, j)
      end
    end
  end
  return src, dst, groups
end

function fbnet(pop, C, n; degreedist)
  src,dst,groups = genedges(pop, C, n; degreedist=degreedist)
  A = sparse(src,dst,1,n,n)
  A = max.(A,A')
  fill!(A.nzval,1)

  xy = xyC[groups,:] # generate xy coords
  return A, groups, xy
end
