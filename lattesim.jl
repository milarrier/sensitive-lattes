using LinearAlgebra
using ToeplitzMatrices
using DifferentialEquations
using Tullio
using FFTW
using Plots

"fft analysis"
function plotticef(t,v)
    ll = length(t)
    N = size(v,2)
    w = 2π*100*(1:ll-2)/ll
    vf = fft(v)
    plot(w, abs.(vf[2:end-1,:]);
         palette=palette(:Blues,N),
         xticks=[0.1,1,10,100],
         xscale=:log10,
         yscale=:log10,
         legend=false)
end

"visual effect of varying interspacing"
function plottice(t,v,delta=1.0)
    (nt, N) = size(v)
    vr = repeat(delta*collect(N-1:-1:0),1,nt)'
    plot(t, v+vr; palette = palette(:Blues,N), legend=false)
end

function simlattice(N)
    A,B,C = proctrl('p')
    n = size(A,1)
    Z0 = fill(0.0,(n+1,1,N,N))
    tspan = (0.0,314.0)
    
    prob = RODEProblem(rhslattice, Z0, tspan; rand_prototype = zeros(N,N))
    sol = solve(prob, RandomEM(), save_noise=true, dt=1/100)

    nt = length(sol.t)
    V = [Array{Float64,2}(undef,N,N) for _ = 1:nt]
    for t in 1:nt
        X = sol.u[t][1:n,:,:,:]
        @tullio Vt[i,k,l,m] := C[i,j] * X[j,k,l,m]
        V[t] = Vt[1,1,:,:] + sol.W[t]
    end
    VV = Array{Float64,2}(undef, nt, N)
    for t in 1:nt
        VV[t,:] = V[t][:,div(N+1,2)]
    end
    return sol.t, VV
end

function rhslattice(dZ, Z, p, t, W)
    A,B,C = proctrl('p')
    Ak,Bk,Ck,Dk = proctrl('c')
    N = size(Z,3)
    n = size(A,1)
    L = laplatticy(N)

    X = Z[1:n,:,:,:] # nx1xNxN
    Ξ = Z[end,1,:,:] # NxN
    @tullio V[i,k,l,m] := C[i,j] * X[j,k,l,m] # 1x1xNxN = 1xn * nx1xNxN
    dV = reshape(V,N,N) + W # NxN
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

    prob = RODEProblem(rhstring, Z0, tspan; rand_prototype = zeros(N))
    sol = solve(prob, RandomEM(), save_noise=true, dt=1/100)

    nt = length(sol.t)
    V = [Array{Float64,3}(undef,1,1,N) for _ = 1:nt]
    for t in 1:nt
        X = sol.u[t][1:n,:,:]
        @tullio V[t][i,k,l] = C[i,j] * X[j,k,l]
    end
    VV = Array{Float64,2}(undef,nt,N)
    for t in 1:nt
        VV[t,:] = hvcat(1, V[t]...) + sol.W[t]
    end
    Vr = repeat(collect(N-1:-1:0),1,nt)'
    plot(sol.t, VV+Vr; palette = palette(:Blues, N), legend=false)
    # return sol.t, VV
end

function rhstring(dZ,Z,p,t,W)
    A,B,C = proctrl('p')
    Ak,Bk,Ck,Dk = proctrl('c')
    n = size(A,1)
    N = size(Z,3)
    Bin = incstr(N)
    
    X = Z[1:n,:,:] # nx1xN
    Ξ = Z[end,1,:] # Nx1
    @tullio V[i,k,l] := C[i,j] * X[j,k,l] # 1x1xN = 1xn * nx1xN
    dV = [V...] + W # Nx1
    Ve = Bin*dV # Nx1
    Je = Ck*Ξ + Dk*Ve # Nx1
    J = reshape(-Bin'*Je, 1,1,N)
    @tullio dX[i,k,l] := A[i,j] * X[j,k,l] # nx1xN = nxn * nx1xN
    @tullio dX[i,k,l] += B[i,j] * J[j,k,l] # nx1xN = nx1 * 1x1xN
    dΞ = Ak*Ξ + Bk*Ve # Nx1
    dZ[1:n,:,:] = dX
    dZ[end,1,:] = dΞ
end

"incidence matrix B: B'B=-L"
function incstr(N)
    if N > 1
        vc = [1;-1;fill(0,N-2)]
        vr = [1;fill(0,N-1)] # acyclic
        # vr = [1;fill(0,N-2);-1] # cyclic
        B = Toeplitz(vc, vr)
    else
        B = reshape(Int64[],N,0)
    end
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
