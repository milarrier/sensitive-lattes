using LinearAlgebra
using ToeplitzMatrices
using DifferentialEquations
using Tullio
using Plots

function simCLotus(N)
    C = [1 0 0]
    Ck = -780
    Dk = 40
    n = size(C,2)
    Z0 = repeat(reshape(transpose([collect(N-1:-1:0) zeros(N,n)]), n+1, 1, N), 1, 1, 1, N)
    L = laplettuce(N) # -lapstr(N) #laplettuce(N)
    # L[1,1] = -1
    # L[end,end] = -1
    tspan = (0.0,100.0)
    prob = RODEProblem(rhsCLotus!, Z0, tspan, rand_prototype = zeros(N,N))
    sol = solve(prob, RandomEM(), dt = 1 / 100)
    nt = length(sol.t)
    V = [Array{Float64,4}(undef,1,1,N,N) for _ = 1:nt]
    for t in 1:nt
        X = sol.u[t][1:n,:,:,:]
        @tullio V[t][i,k,l,m] = C[i,j] * X[j,k,l,m]
    end
    VV = Array{Float64,2}(undef, nt, N^2)
    for t in 1:nt
        VV[t,:] = hvcat(N^2, V[t]...)
    end
    # J = Array{Float64,2}(undef, nt, N^2)
    # for t in 1:nt
    #     Ξ = [sol.u[t][end,:,:,:]...]
    #     J[t,:] = Ck*Ξ + Dk*VV[t,:]
    # end
    # U = Array{Float64,2}(undef, nt, N^2)
    # for t in 1:nt
    #     JJ = reshape(J[t,:], N, N)
    #     U[t,:] = reshape(L*JJ + JJ*L, N^2, 1)
    # end
    # p1 = plot(sol.t, VV[:,1:N]; palette = palette(:Blues, rev=true), legend = false)
    # p2 = plot(sol.t, J[:,1:N]; palette = palette(:Blues, rev=true), legend = false)
    # p3 = plot(sol.t, U[:,1:N]; palette = palette(:Blues, rev=true), legend = false)
    # plot(p1, p2, p3, layout=(3,1))
    plot(sol.t, VV[:,floor(Int,N/2)*N+1:ceil(Int,N/2)*N]; palette = palette(:Blues), legend = false)
end

function rhsCLotus!(dZ, Z, p, t, W)
    a = 0.1
    A = [0 1 0; 0 0 1; 0 0 -1/a]
    B = reshape([0 0 1/a], size(A,1), 1)
    C = [1 0 0]
    Ak = -20
    Bk = 1
    Ck = -780
    Dk = 40
    N = size(Z,3)
    n = size(A,1)
    L = laplettuce(N) #-lapstr(N) #laplettuce(N)
    # L[1,1] = -1
    # L[end,end] = -1
    Vr = repeat(collect(N-1:-1:0), 1, N)
    maskW = zeros(N,N)
    imid = ceil(Int,N/2)
    maskW[imid,imid] = 1
    X = Z[1:n,:,:,:] # nx1xNxN
    Ξ = Z[end,1,:,:] # NxN
    @tullio V[i,k,l,m] := C[i,j] * X[j,k,l,m] # 1x1xNxN = 1xn * nx1xNxN
    dV = reshape(V,N,N) + 0.01*maskW.*W # NxN
    J = Ck*Ξ + Dk*dV - Vr # NxN
    U = reshape(L*J + J*L, 1,1,N,N) # 1x1xNxN
    @tullio dX[i,k,l,m] := A[i,j] * X[j,k,l,m] # nx1xNxN = nxn * nx1xNxN
    @tullio dX[i,k,l,m] += B[i,j] * U[j,k,l,m] # nx1xNxN = nx1 * 1x1xNxN
    dΞ = Ak*Ξ + Bk*dV
    dZ[1:n,:,:,:] = dX
    dZ[end,1,:,:] = dΞ
end

function simstring(N)
    C = [1 0 0]
    n = size(C,2)
    Z0 = reshape(transpose([collect(N-1:-1:0) zeros(N,n)]), n+1, 1, N)
    L = -lapstr(N)
    tspan = (0.0,41943.04)
    prob = RODEProblem(rhstring, Z0, tspan, L; rand_prototype = zeros(N), save_noise=true)
    sol = solve(prob, RandomEM(), dt = 1 / 100)
    nt = length(sol.t)
    V = [Array{Float64,3}(undef,1,1,N) for _ = 1:nt]
    for t in 1:nt
        X = sol.u[t][1:n,:,:]
        @tullio V[t][i,k,l] = C[i,j] * X[j,k,l]
    end
    VV = Array{Float64,2}(undef, nt, N)
    for t in 1:nt
        VV[t,:] = hvcat(N, V[t]...)
    end
    # plot(sol.t, VV; palette = palette(:Blues, rev=true), legend = false)
    return sol, VV
end

function rhstring(dZ, Z, L, t, W)
    a = 0.1
    A = [0 1 0; 0 0 1; 0 0 -1/a]
    B = reshape([0 0 1/a], size(A,1), 1)
    C = [1 0 0]
    Ak = -20
    Bk = 1
    Ck = -780
    Dk = 40
    N = size(Z,3)
    n = size(A,1)
    Vr = collect(N-1:-1:0)
    maskW = zeros(N)
    maskW[ceil(Int,N/2)] = 1
    X = Z[1:n,:,:] # nx1xN
    Ξ = Z[end,1,:] # Nx1
    @tullio V[i,k,l] := C[i,j] * X[j,k,l] # 1x1xN = 1xn * nx1xN
    dV = [V...] + 0.01*W .* maskW # Nx1
    J = Ck*Ξ + Dk*dV # Nx1
    U = reshape(L*(J-Vr), 1,1,N) # Nx1 -> 1x1xN
    @tullio dX[i,k,l] := A[i,j] * X[j,k,l] # nx1xN = nxn * nx1xN
    @tullio dX[i,k,l] += B[i,j] * U[j,k,l] # nx1xN = nx1 * 1x1xN
    dΞ = Ak*Ξ + Bk*dV
    dZ[1:n,:,:] = dX
    dZ[end,1,:] = dΞ
end

# function simcaravg(N)
#     C = [1. 0. 0.]
#     nx = size(C,2)
#     L = laplettuce(N)
#     X0 = repeat(reshape(transpose([collect(N-1:-1:0) zeros(N,nx-1)]), nx, 1, N), 1, 1, 1, N)
#     # W = t -> rand(nx,1,N,N)
#     tspan = (0.0,2.0)
#     prob = ODEProblem(rhscaravg!, X0, tspan, L)
#     sol = solve(prob, RK4(), saveat = 0.01, abstol = 1e-9, reltol = 1e-9)
#     nt = length(sol.t)
#     V = [Array{Float64,4}(undef,1,1,N,N) for _ = 1:nt] # amazing comprehension
#     for n in 1:nt
#         @tullio V[n][i,k,l,m] = C[i,j] * sol.u[n][j,k,l,m]
#     end
#     VV = Array{Float64,2}(undef, nt, N^2)
#     for n in 1:nt
#         VV[n,:] = hvcat(N^2, V[n]...)
#     end
#     plot(sol.t, VV[:,2N+1:3N]; palette = palette(:Blues_4, rev=true), legend = false)
# end

# function rhscaravg!(dX, X, L, t)
#     # p = ss(tf(1, [0.1,1,0,0]))
#     a = 0.1
#     A = [0 1 0; 0 0 1; 0 0 -1/a]
#     B = reshape([0 0 1/a], size(A,1), 1)
#     C = [1. 0. 0.]
#     # D = 0.0
#     @tullio V[i,k,l,m] := C[i,j] * X[j,k,l,m] # 1x1xNxN = 1xn * nx1xNxN
#     @tullio U[i,j,m,l] := L[m,k] * V[i,j,k,l] + V[i,j,m,k] * L[k,l] # 1x1xNxN = NxN * 1x1xNxN + 1x1xNxN * NxN
#     @tullio dX[i,k,l,m] = A[i,j] * X[j,k,l,m] # nx1xNxN = nxn * nx1xNxN; crucial in-place = not :=
#     @tullio dX[i,k,l,m] += B[i,j] * U[j,k,l,m] # nx1xNxN = nx1 * 1x1xNxN
# end

# function simlettuce(N)
#     V0 = rand(N,N)
#     tspan = (0.0, 1.0)
#     W = t -> rand(N,N)
#     L = laplettuce(N)
#     rhslettuce(V, W, t) = V*L + L*V + W(t)
#     prob = ODEProblem(rhslettuce, V0, tspan, W)
#     sol = solve(prob)
#     plot(sol)
# end

function laplettuce(N)
    col1 = zeros(N)
    row1 = zeros(N)
    col1[2] = 1.0
    row1[end] = 1.0
    U = Toeplitz(col1, row1)
    L = U + U' - 2I
end

function lapstr(N)
    d = vec(2*ones(N,1))
    d[1] = 1
    dl = vec(-1*ones(N-1,1))
    L = Tridiagonal(dl, d, dl)
end

# Array{Array{}} as state variable not supported by DifferentialEquations
# one-liner functions can access outside variables but block functions can not?
# what is @tullio dX[i,k,l,m] := A[i,j] * X[j,k,l,m] + B[i,jj] * UU[jj,k,l,m] doing?
