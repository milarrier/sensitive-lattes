using LinearAlgebra
using ToeplitzMatrices
using DifferentialEquations
using Tullio
using Plots

function simstring(N)
    C = [1 0 0]
    n = size(C,2)
    # Z0 = reshape(transpose([collect(N-1:-1:0) zeros(N,n)]), n+1, 1, N)
    Z0 = fill(0.0,(n+1,1,N))
    tspan = (0.0,300.0)

    prob = RODEProblem(rhstringjr, Z0, tspan; rand_prototype = zeros(N))
    sol = solve(prob, RandomEM(), dt = 1 / 100)
    # W = t -> sin(0.1382t)
    # prob = ODEProblem(rhstring, Z0, tspan, W)
    # sol = solve(prob, dt = 1/100)

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
    Vr = repeat(collect(N-1:-1:0),1,nt)'
    plot(sol.t, VV+Vr; palette = palette(:Blues, N), legend = false)
    # return sol, VV
end

"output disturbance"
function rhstringr(dZ, Z, p, t, W)
# function rhstring(dZ, Z, W, t)
    A,B,C = proctrl('p')
    Ak,Bk,Ck,Dk = proctrl('c')
    N = size(Z,3)
    n = size(A,1)
    L = -lapstr(N)
    # Vr = collect(N-1:-1:0)
    maskW = ones(N)
    # maskW[ceil(Int,N/2)] = 1
    
    X = Z[1:n,:,:] # nx1xN
    Ξ = Z[end,1,:] # Nx1
    @tullio V[i,k,l] := C[i,j] * X[j,k,l] # 1x1xN = 1xn * nx1xN
    dV = [V...] + 0.01W .* maskW # output disturbance random Nx1
    # dV = [V...] + W(t) .* maskW # output disturbance deterministic Nx1
    J = Ck*Ξ + Dk*dV # Nx1
    # U = reshape(L*(J-Vr), 1,1,N) # with reference Nx1 -> 1x1xN
    U = reshape(L*J, 1,1,N)
    @tullio dX[i,k,l] := A[i,j] * X[j,k,l] # nx1xN = nxn * nx1xN
    @tullio dX[i,k,l] += B[i,j] * U[j,k,l] # nx1xN = nx1 * 1x1xN
    dΞ = Ak*Ξ + Bk*dV
    dZ[1:n,:,:] = dX
    dZ[end,1,:] = dΞ
end

"output disturbance delayed on J"
function rhstringjr(dZ, Z, p, t, W)
# function rhstringj(dZ, Z, W, t)
    A,B,C = proctrl('p')
    Ak,Bk,Ck,Dk = proctrl('c')
    N = size(Z,3)
    n = size(A,1)
    L = -lapstr(N)
    # Vr = collect(N-1:-1:0)
    maskW = zeros(N)
    maskW[ceil(Int,N/2)] = 1
    
    X = Z[1:n,:,:] # nx1xN
    Ξ = Z[end,1,:] # Nx1
    @tullio V[i,k,l] := C[i,j] * X[j,k,l] # 1x1xN = 1xn * nx1xN
    dV = [V...] + 0.01W .* maskW # output disturbance random Nx1
    J = Ck*Ξ + Dk*[V...] # Nx1
    # U = reshape(L*(J-Vr), 1,1,N) # with reference Nx1 -> 1x1xN
    U = reshape(L*J, 1,1,N)
    @tullio dX[i,k,l] := A[i,j] * X[j,k,l] # nx1xN = nxn * nx1xN
    @tullio dX[i,k,l] += B[i,j] * U[j,k,l] # nx1xN = nx1 * 1x1xN
    dΞ = Ak*Ξ + Bk*dV
    dZ[1:n,:,:] = dX
    dZ[end,1,:] = dΞ
end

"disturb U"
function rhstringur(dZ, Z, p, t, W)
# function rhstringu(dZ, Z, W, t)
    A,B,C = proctrl('p')
    Ak,Bk,Ck,Dk = proctrl('c')
    N = size(Z,3)
    n = size(A,1)
    L = -lapstr(N)
    maskW = zeros(N)
    maskW[end] = 1
    
    X = Z[1:n,:,:] # nx1xN
    Ξ = Z[end,1,:] # Nx1
    @tullio V[i,k,l] := C[i,j] * X[j,k,l] # 1x1xN = 1xn * nx1xN
    dV = [V...]
    J = Ck*Ξ + Dk*dV # Nx1
    U = reshape(L*J + 0.01W .* maskW, 1,1,N) # input disturbance random
    # U = reshape(L*J + W(t) .* maskW, 1,1,N) # input disturbance
    @tullio dX[i,k,l] := A[i,j] * X[j,k,l] # nx1xN = nxn * nx1xN
    @tullio dX[i,k,l] += B[i,j] * U[j,k,l] # nx1xN = nx1 * 1x1xN
    dΞ = Ak*Ξ + Bk*dV
    dZ[1:n,:,:] = dX
    dZ[end,1,:] = dΞ
end

function lapstr(N)
    d = vec(2*ones(N,1))
    d[end] = 1
    dl = vec(-1*ones(N-1,1))
    L = Tridiagonal(dl, d, dl)
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
