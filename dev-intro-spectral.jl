

##
using SparseArrays, MatrixNetworks, Printf, LinearAlgebra
import LinearAlgebra.checksquare
function _compute_eigen(A::SparseMatrixCSC,d::Vector,nev::Int,
        tol=1e-12,maxiter=300,dense::Int=96)

    n = checksquare(A)
    if n == 1
        X = zeros(typeof(one(eltype(d))/one(eltype(d))),1,nev)
        lams = [0.0]
    elseif n <= dense
        ai,aj,av = findnz(A)
        L = sparse(ai,aj,-av./((d[ai].*d[aj])),n,n) # applied sqrt above
        L = Matrix(L) + 2I
        F = eigen!(Symmetric(L))
        lams = F.values.-1.0
        X = F.vectors
    else # modifying this branch 1-11-2019
        ai,aj,av = findnz(A)
        L = sparse(ai,aj,-av./((d[ai].*d[aj])),n,n) # applied sqrt above
        L = L + sparse(2.0I,n,n) # Rich Lehoucq suggested this idea to
                                 # make ARPACKs internal tolerance better scaled.

        (lams,X,nconv) = _symeigs_smallest_arpack(L,nev,tol,maxiter,d)
        lams = lams.-1.0
    end
    X = X[:,sortperm(lams)] # sort in ascending order
    if size(X,2) > nev
        X = X[:,1:nev]
        lams = lams[1:nev]
    end
    return X, lams
end


""" `spectral_embedding(A,k)`

Get a spectral embedding of a sparse matrix A that represents a graph.

This handles small matrices by using LAPACK and large sparse matrices with
ARPACK. Given a sparse matrix \$ A \$, this returns the smallest eigenspace
of the generalized eigenvalue problem min x'*L*x/x'*D*x where L is the
Laplacian and D is the degree matrix.  The sign of the eigenspace
is based on the vertex with maximum degree.

## Inputs
- `A::SparseMatrix` A sparse matrix that represents the graph data.
- `k::Int` the number of eigenvectors to compute
## Outputs
- `X::Matrix` A k-column marix where each column are the eigenvectors and X[:,1]
  is the standard null-space vector. The sign of each column is picked by choosing
  nodes with maximum degree to have positive signs.
## Optional inputs
- `normalize::Bool=true` produce the degree-normalized generalized eigenvectors of
      D^{-1} L (normalize=true) instead of the normalized
      Laplacian D^{-1/2} L D^{-1/2} (normalize=false)
## Uncommon Inputs (where defaults should be handled with care.)
- `dense::Int` the threshold for a dense (LAPACK) computation
- `checksym::Bool` A switch to turn off symmetry checking (don't do this)
- `tol::Real` A real-valued tolerance to give to ARPACK
- `maxiter::Int` The maximum number of iterations for ARPACK
"""
function spectral_embedding(A::SparseMatrixCSC{V,Int},k::Int;
        tol=1e-12,maxiter=300,dense::Int=96,checksym=true,
        normalize::Bool=true) where {V <: Real}
# ### History
# This code is from the GLANCE package, based on code from the
# MatrixNetworks.jl package, based on a Matlab spectral clustering code.
# also from my diffusion-tools.jl code from another package.

    n = checksquare(A)
    if checksym
        if !issymmetric(A)
            throw(ArgumentError("The input matrix must be symmetric."))
        end
    end

    nev = k
    d = vec(sum(A,dims=1))
    d = sqrt.(d)

    X,lams = _compute_eigen(A,d,nev,tol,maxiter,dense)

    x1err = norm(X[:,1]*sign(X[1,1]) - d/norm(d))
    if x1err >= sqrt(tol)
        s = @sprintf("""
        the null-space vector associated with the normalized Laplacian
        was computed inaccurately (diff=%.3e); the Fiedler vector is
        probably wrong or the graph is disconnected""",x1err)
        @warn s
    end

    vdmax = argmax(d)  # the vertex of maximum degree
    X ./= repeat(d,1,size(X,2))
    X .*= repeat(sign.(X[1,:])',size(X,1),1)
    #X .*= sign(X[1])

    return X, lams
end

function cycle_graph(n::Int)
  A = sparse(1:n-1,2:n,1,n,n)
  A[1,end] = 1
  A = max.(A,A')
  return A
end

Random.seed!(0)
#A = cycle_graph(10)
A = sprand(10,10,3/10)
A = max.(A,A')
fill!(A.nzval, 1.0)
A .-= Diagonal(A)
X = spectral_embedding(A,3)[1][:,2:end]
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
F = simple_spectral_eigenvectors(A,2)
##
norm(X-F)
##
## Testing plotting by rank...
byrank(x) = invperm(sortperm(x))
R = hcat(map(byrank,eachcol(F))...)
plot(graph_lines(A,R)...,marker=:dot,
  markersize=1,markerstrokewidth=0,linewidth=0.25,linealpha=0.05,markeralpha=0.1,
  framestyle=:none,legend=false)
