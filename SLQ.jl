using LinearAlgebra
using SparseArrays

module SLQ

using SparseArrays, DataStructures, LinearAlgebra, ProgressMeter
struct GraphAndDegrees{
        T<: Union{Float32,Float64,Int32,Int64},
        Ti <: Union{Int,Int32,Int64}}   # T is the type of edges,
  A::SparseMatrixCSC{T,Ti}
  deg::Vector{T}
end

abstract type EdgeLoss{T} end

struct QHuberLoss{T} <: EdgeLoss{T}
    q::T
    delta::T
end

struct TwoNormLoss{T} <: EdgeLoss{T}
end

""" This function isn't type stable, so don't use it except it outer codes. """
function loss_type(q::T, delta::T) where T
    if q == 2.0
        return TwoNormLoss{T}()
    else
        return QHuberLoss{T}(q, delta)
    end
end

minval(f, L::QHuberLoss) = f^(1/(L.q-1))
minval(f, L::TwoNormLoss) = sqrt(f)

function loss_gradient(x::T, L::QHuberLoss{T}) where T
    if abs(x) < L.delta
        return L.delta^(L.q-2)*x
    else
        return sign(x)*(abs(x)^(L.q-1))
    end
end

function loss_gradient(x::T, L::TwoNormLoss{T}) where T
    return x
end

function loss_function(x::T, L::QHuberLoss{T}) where T
    if abs(x) < L.delta
        return 0.5*(L.delta^(L.q-2))*(x^2)
    else
        return (abs(x)^L.q)/L.q+(0.5-1/L.q)*(L.delta^L.q)
    end
end

function loss_function(x::T, L::TwoNormLoss{T}) where T
    return 0.5*(x^2)
end

function graph(A::SparseMatrixCSC)
    d = vec(sum(A,dims=2))
    return GraphAndDegrees(A, d)
end

function _buffer_neighbors!(x::Vector, A::SparseMatrixCSC,
        i::Int, buf_x::Vector{T}, buf_vals::Vector{T}) where T
    nneighs = A.colptr[i+1]-A.colptr[i]
    for (iter,k) in enumerate(A.colptr[i]:(A.colptr[i+1]-1))
        j = A.rowval[k]
        buf_x[iter] = x[j]
        buf_vals[iter] = T(A.nzval[k])
    end
    return nneighs
end


function _eval_residual_i(xi::T, di::T, dx::T, seed::Bool,
        neigh_x::AbstractVector{T}, neigh_vals::AbstractVector{T},
        L::EdgeLoss{T}, gamma::T) where T

    ri_new = zero(T)
    for k in 1:length(neigh_x)
      ri_new -= neigh_vals[k]*loss_gradient(xi+dx-neigh_x[k],L)/gamma
    end
    if seed
      ri_new -= di*loss_gradient(xi+dx-1,L)
    else
      ri_new -= di*loss_gradient(xi+dx,L)
    end
    return ri_new
end

function dxi_solver(G::GraphAndDegrees,x::Vector{T},
        kappa::T,epsilon::T,gamma::T,r::Vector{T},
        seedset,rho::T,i::Int,L::TwoNormLoss{T},
        buf_x::Vector,buf_vals::Vector,thd1,thd2) where T

    di = G.deg[i]
    found_dxi = false
    A = G.A

    dxi = r[i]*rho*gamma/(di*(1+gamma))

    return dxi
end

function dxi_solver(G::GraphAndDegrees,x::Vector{T},
        kappa::T,epsilon::T,gamma::T,r::Vector{T},
        seedset,rho::T,i::Int,L::EdgeLoss{T},
        buf_x::Vector,buf_vals::Vector,thd1,thd2) where T
    di = G.deg[i]
    found_dxi = false
    A = G.A
    nneighs::Int = _buffer_neighbors!(x,A,i,buf_x, buf_vals)

    nbisect = 0

    ri_new = r[i]
    dx_min = 0
    thd_min = min(thd1,thd2)
    thd_max = max(thd1,thd2)
    thd = thd_max
    dx = thd
    ri_new = _eval_residual_i(x[i], T(di), dx, i in seedset,
        @view(buf_x[1:nneighs]), @view(buf_vals[1:nneighs]),
        L, gamma)
    if ri_new < 0
        ri_new = r[i]
        thd = thd_min
    end
    last_dx = 0

    ratio = 10 # 2020-05-27 switched this ratio from 2 to 10
    while ri_new > rho*kappa*di
        dx = thd
        ri_new = _eval_residual_i(x[i], T(di), dx, i in seedset,
            @view(buf_x[1:nneighs]), @view(buf_vals[1:nneighs]),
            L, gamma)

        #=
        if nbisect >= 40
            @show i, dx, T(di), ri_new, rho*kappa*di
        end
        =#
        last_dx = dx_min
        dx_min = thd
        thd *= ratio
        nbisect += 1
    end
    dx_min = last_dx
    dx_max = thd/ratio

    dx_mid = 0
    while (found_dxi == false && dx_max - dx_min > epsilon) || (ri_new < 0)
        dx_mid = dx_max/2+dx_min/2
        ri_new = _eval_residual_i(x[i], T(di), dx_mid, i in seedset,
            @view(buf_x[1:nneighs]), @view(buf_vals[1:nneighs]),
            L, gamma)

        if ri_new < rho*kappa*di
            dx_max = dx_mid
        elseif ri_new > rho*kappa*di
            dx_min = dx_mid
        else
            found_dxi = true
        end
    end
    if dx_mid == 0
        dxi = dx_max
    else
        dxi = dx_mid
    end
    return dxi
end

function residual_update!(G::GraphAndDegrees,
        x::Vector,dxi,i,seedset::Set{Int},r,gamma,Q,kappa,L::EdgeLoss)
    A = G.A
    r[i] = 0
    for k in A.colptr[i]:(A.colptr[i+1]-1)
        j = A.rowval[k]
        dri = loss_gradient(x[j]-x[i]-dxi,L)
        drij = A.nzval[k]*(loss_gradient(x[j]-x[i],L)-dri)
        drij /= gamma
        rj_old = r[j]
        r[j] += drij
        r[i] += A.nzval[k]*dri/gamma
        if rj_old <= kappa*G.deg[j] && r[j] > kappa*G.deg[j]
            push!(Q,j)
        end
    end
    if i in seedset
        r[i] -= G.deg[i]*loss_gradient(x[i]+dxi-1,L)
    else
        r[i] -= G.deg[i]*loss_gradient(x[i]+dxi,L)
    end
    if r[i] > kappa*G.deg[i]
        push!(Q,i)
    end
end

function _max_nz_degree(A::SparseMatrixCSC)
    n = A.n
    maxd = zero(eltype(A.colptr))
    for i=1:n
        maxd = max(maxd, A.colptr[i+1]-A.colptr[i])
    end
    return maxd
end

"""
EdgeLoss{T} includes either TwoNormLoss or QHuberLoss, where we have
- `q` the value of q in the q-norm
- `delta` the value of delta in the q-Huber function
use loss_type(q,delta) for a type-unstable solution that will dispatch correctly

- `gamma` is for regularization, Infty returns seed set, 0 is hard/ill-posed.
- `kappa` is the sparsity regularilzation term.
- `rho` is the slack term in the KKT conditions to get faster convergence.
    (rho=1 is slow, rho=0)
- `eps` the value of epsilon in the local binary search
"""
function slq_diffusion(G::GraphAndDegrees,S,gamma::T,kappa::T,rho::T,L::EdgeLoss{T};
        max_iters::Int=1000,epsilon::T=1.0e-8,progress::Bool=true) where {T <: Real}

    A = G.A
    n = size(A,1)
    x = zeros(n)
    r = zeros(n)

    max_deg = _max_nz_degree(A)

    buf_x = zeros(max_deg)
    buf_vals = zeros(max_deg)
    Q = CircularDeque{Int}(n)
    #
    for i in S
        r[i] = G.deg[i]
        push!(Q,i)
    end
    seedset = Set(S)

    iter = 0

    t0 = time()
    checkinterval = 10^5
    if progress == false
        checkinterval = max_iters
    end
    pushvol = 0
    nextcheck = checkinterval
    notify_time = 60.0
    last_time = t0
    last_iter = 0
    used_pm = false
    pm = Progress(max_iters, "SLQ: ")

    #thd1 = (sum(G.deg[S])/sum(G.deg))^(1/(q-1))
    thd1 = minval(sum(G.deg[S])/sum(G.deg), L)
    thd2 = thd1


    while length(Q) > 0 && iter < max_iters
        i = popfirst!(Q)
        dxi = dxi_solver(G,x,kappa,epsilon,gamma,r,seedset,rho,i,L,buf_x,buf_vals,thd1,thd2)
        thd2 = dxi
        residual_update!(G,x,dxi,i,seedset,r,gamma,Q,kappa,L)
        x[i] += dxi

        pushvol += A.colptr[i+1] - A.colptr[i]
        iter += 1

        if iter > nextcheck
            nextcheck = iter+checkinterval
            ct = time()

            if ct - t0 >= notify_time
                used_pm = true
                ProgressMeter.update!(pm, iter; showvalues =
                    [(:pushes_per_second,(iter-last_iter)/(ct-last_time)),
                     (:edges_per_second,pushvol/(ct-last_time))])
            end

            last_iter = iter
            last_time = ct
            pushvol = 0
        end
    end

    if used_pm == true
        ProgressMeter.finish!(pm)
    end

    if iter == max_iters && length(Q) > 0
        @warn "reached maximum iterations"
    end
    return x,r,iter
end


function objective(G::GraphAndDegrees,S,x::Vector{T},
        kappa::Real,gamma::Real,L::EdgeLoss{T}) where T
    obj = 0.0
    A = G.A
    n = size(A,1)
    for i in 1:n
        for k in A.colptr[i]:(A.colptr[i+1]-1)
            j = A.rowval[k]
            obj += A.nzval[k]*loss_function(x[i]-x[j],L)
        end
    end
    for i in S
        obj += gamma*G.deg[i]*loss_function(x[i]-1,L)
    end
    Sbar = setdiff(1:n,S)
    for i in Sbar
        obj += gamma*G.deg[i]*loss_function(x[i],L)
    end
    obj += kappa*gamma*sum(G.deg.*x)
    return obj
end

end # end module

#=
include("common.jl")
using Test
@testset "SLQ" begin
    A,xy = two_cliques(5,5)
    @test_nowarn SLQ.slq_diffusion(SLQ.graph(A), [1], 0.1, 0.1, 0.5,
        SLQ.loss_type(2.0,0.0))
    @test_nowarn SLQ.slq_diffusion(SLQ.graph(A), [1], 0.1, 0.1, 0.5,
        SLQ.QHuberLoss(2.0,0.0))

    A = sparse(ones(10,10)-I)
    G = SLQ.graph(A)
    x, r, iters = SLQ.slq_diffusion(G, [1], 0.1, 1.0, 0.99999,
        SLQ.loss_type(2.0, 0.0))
    @test all(isfinite.(r))
end
=#
