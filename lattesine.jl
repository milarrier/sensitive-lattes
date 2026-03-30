using LinearAlgebra
using ToeplitzMatrices
using DifferentialEquations
using Tullio
using Plots

function simlattice(N)
    A,B,C = proctrl('p')
    n = size(A,1)
    Z0 = fill(0.0,(n+1,1,N,N))
    tspan = (0.0,314.0)

    W = t -> sin(0.1t)
    prob = ODEProblem(rhslattice, Z0, tspan, W)
    sol = solve(prob, dt=1/100)

    nt = length(sol.t)
    V = [Array{Float64,4}(undef,1,1,N,N) for _ = 1:nt]
    for t in 1:nt
        X = sol.u[t][1:n,:,:,:]
        @tullio V[t][i,k,l,m] = C[i,j] * X[j,k,l,m]
    end
    VV = Array{Float64,1}(undef, nt)
    m = div(N+1,2)
    for t in 1:nt
        VV[t] = V[t][1,1,m,m] + W(sol.t[t])
    end
    return sol.t, VV
end

function rhslattice(dZ, Z, W, t)
    A,B,C = proctrl('p')
    Ak,Bk,Ck,Dk = proctrl('c')
    N = size(Z,3)
    n = size(A,1)
    L = laplatticy(N)

    X = Z[1:n,:,:,:] # nx1xNxN
    Ξ = Z[end,1,:,:] # NxN
    @tullio V[i,k,l,m] := C[i,j] * X[j,k,l,m] # 1x1xNxN = 1xn * nx1xNxN
    dV = reshape(V,N,N) + W(t)*maskW(N,2) # NxN
    J = Ck*Ξ + Dk*dV # NxN
    U = reshape(L*J + J*L, 1,1,N,N) # 1x1xNxN
    @tullio dX[i,k,l,m] := A[i,j] * X[j,k,l,m] # nx1xNxN = nxn * nx1xNxN
    @tullio dX[i,k,l,m] += B[i,j] * U[j,k,l,m] # nx1xNxN = nx1 * 1x1xNxN
    dΞ = Ak*Ξ + Bk*dV
    dZ[1:n,:,:,:] = dX
    dZ[end,1,:,:] = dΞ
end

function simstring(N)
    A,B,C = proctrl('p')
    n = size(A,1)
    Z0 = fill(0.0,(n+1,1,N))
    tspan = (0.0,300.0)

    W = t -> sin(0.1382t)
    prob = ODEProblem(rhstring, Z0, tspan, W)
    sol = solve(prob, dt = 1/100)

    nt = length(sol.t)
    V = [Array{Float64,3}(undef,1,1,N) for _ = 1:nt]
    for t in 1:nt
        X = sol.u[t][1:n,:,:]
        @tullio V[t][i,k,l] = C[i,j] * X[j,k,l]
    end
    VV = Array{Float64,1}(undef, nt)
    for t in 1:nt
        VV[t] = V[t][1,1,div(N+1,2)] + W(sol.t[t])
    end
    return sol.t, VV
end

function rhstring(dZ, Z, W, t)
    A,B,C = proctrl('p')
    Ak,Bk,Ck,Dk = proctrl('c')
    N = size(Z,3)
    n = size(A,1)
    # Bin = incstr(N)
    L = laplattica(N)
    
    X = Z[1:n,:,:] # nx1xN
    Ξ = Z[end,1,:] # Nx1
    @tullio V[i,k,l] := C[i,j] * X[j,k,l] # 1x1xN = 1xn * nx1xN
    dV = [V...] + W(t)*maskW(N,1) # output disturbance deterministic Nx1
    # Ve = Bin*dV # Nx1
    # Je = Ck*Ξ + Dk*Ve # Nx1
    # J = reshape(-Bin'*Je, 1,1,N)
    Je = Ck*Ξ + Dk*dV
    J = reshape(L*Je, 1,1,N)
    @tullio dX[i,k,l] := A[i,j] * X[j,k,l] # nx1xN = nxn * nx1xN
    @tullio dX[i,k,l] += B[i,j] * J[j,k,l] # nx1xN = nx1 * 1x1xN
    # dΞ = Ak*Ξ + Bk*Ve # Nx1
    dΞ = Ak*Ξ + Bk*dV
    dZ[1:n,:,:] = dX
    dZ[end,1,:] = dΞ
end

"cyclic laplacian"
function laplatticy(N)
    col1 = zeros(N)
    row1 = zeros(N)
    col1[2] = 1.0
    row1[end] = 1.0
    U = Toeplitz(col1, row1)
    L = U + U' - 2I
end

"acyclic laplacian"
function laplattica(N)
    d = fill(2,N)
    d[end] = 1
    dl = fill(-1,N-1)
    L = -Tridiagonal(dl, d, dl)
end

"acyclic incidence matrix"
function incstr(N)
    if N > 1
        vc = [1;-1;fill(0,N-2)]
        vr = [1;fill(0,N-1)]
        B = Toeplitz(vc, vr)
    else
        B = reshape(Int64[],N,0)
    end
end

function proctrl(pc::Char, numex::Int=1)
    if numex == 1
        if pc == 'p'
            A = [0 1 0; 0 0 1; 0 0 -10.0]
            B = reshape([0 0 10.0], size(A,1), 1)
            C = [1 0 0]
            D = 0
        elseif pc == 'c'
            A = -20
            B = 1
            C = -780
            D = 40
        else
            error("enter either 'p' or 'c' svp")
        end
    end
    return A,B,C,D
end

function maskW(N::Int, d::Int)
    mask1 = fill(0,N)
    mask1[div(N+1,2)] = 1
    if d==1
        mask = mask1
    elseif d==2
        mask = mask1*mask1'
    else
        error("i dont care for d>2")
    end
    return mask
end
