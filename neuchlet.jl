using LinearAlgebra
using ControlSystems
using FFTW
using Plots

"simulates in freq domain then converts to time domain"
function simNeuletF(N::Int64, tend::Float64=50.0)
    nt = 2^23
    nt2 = div(nt,2)
    tpad = 1024*tend
    dt = tpad/nt
    dw = 2π/tpad
    t = dt*(0:nt-1)
    w = dw*(-nt2:nt2-1)
    it = floor(Int, tend/dt)
    v00 = zeros(nt)
    # p = plot(layout=(N,N))
    # for (m,n) in collect(Iterators.product(1:N,1:N))
    vhat = frSNeulet(N,(1,1),(N,N),w)
    uu = vcat(randp(nt2), zeros(nt2))
        uw = fft(uu) # fft(vcat(randl(t[1:nt2]), zeros(nt2)))
        v = real(ifft(fftshift(vhat).*uw))
        v00 += v
        # plot!(t[1:it],v[1:it], subplot=N*(m-1)+n, legend=false)
        # display(p)
    # end
    return t[1:it], v00[1:it], uu[1:it]
    # return p
end

"frequency response from (k,l) to (m,n) obtained node-wise and then summed up"
function frSNeulet(N::Int64, (k,l)::Tuple{Int64,Int64}, (m,n)::Tuple{Int64,Int64}, w)
    nw = length(w)
    r = zeros(ComplexF64, nw, Threads.nthreads())
    p = tf(1, [0.1,1,0,0])
    c = tf([2,1], [0.05,1])
    # p = tf(1, [1,1])
    # c = tf(1, [1,0])
    g = p*c
    θ = π/(2N+1)
    Threads.@threads for (i,j) in collect(Iterators.product(1:N,1:N))
        σij = sin((2i-1)θ/2)^2+sin((2j-1)θ/2)^2
        Sij = sin((2i-1)k*θ)*sin((2j-1)l*θ)*sin((2i-1)m*θ)*sin((2j-1)n*θ)/(1+4g*σij)
        r[:,Threads.threadid()] += dropdims(freqresp(Sij,w); dims=(1,2))
    end
    r = sum(r, dims=2)
    # iw0 = findall(iszero, w) # at zero frequency it's simply zero in this case
    r = 16r/(2N+1)^2
end

#=============================== SANITY CHECKS ================================#
"compares simNeuletF result with time domain simulation for small N"
function pltSimNeulet(N::Int64, tend::Float64=81.92)
    tw, vw, uw = simNeuletF(N,tend)
    p = plot(layout=(2,1))
    plot!(tw, uw; subplot=1)
    plot!(tw, vw; subplot=2)
    dt = 0.01
    t = 0:dt:tend
    u = randp(length(t)) # when using randp() tend=dt*nt/1024
    v,tout,x,uout = lsim(SNeulet(N,(1,1),(N,N)), u', t)
    plot!(t, u; line=:dot, subplot=1)
    plot!(t, v[1,:]; line=:dot, subplot=2)
    return p
end

function randp(nt)
    u = zeros(nt)
    for i = 1:nt
        u[i] = randn(Xoshiro(i))
    end
    return u
end
# randp(nt) = randn(Xoshiro(1),nt)
# some batch size shit prevents me from getting the same first n1 elements
# when comparing randp(n1) and randp(n2) for n1<n2

# ufn(t) = sin.(π/5*t)
# ufn(t) = sin.(t)

"finite sensitivity function in state-space form"
function SNeulet(N::Int64, (k,l)::Tuple{Int64,Int64}, (m,n)::Tuple{Int64,Int64})
    A = [0 1 0; 0 0 1; 0 0 -10]
    B = [0; 0; 10]
    C = [1 0 0]
    Ak = -20
    Bk = 1
    Ck = -780
    Dk = 40
    p = ss(A,B,C,0.0)
    c = ss(Ak,Bk,Ck,Dk)
    g = minreal(p*c)
    θ = π/(2N+1)
    S = 0
    for (i,j) in collect(Iterators.product(1:N,1:N))
        σij = sin((2i-1)θ/2)^2+sin((2j-1)θ/2)^2
        S += sin((2i-1)k*θ)*sin((2j-1)l*θ)*sin((2i-1)m*θ)*sin((2j-1)n*θ)/(1+4g*σij)
    end
    return minreal(S)*16/(2N+1)^2
end

# "random noise passed through an LPF"
# function randl(t)
#     nt = length(t)
#     u = randn(nt)
#     s = tf("s")
#     out = lsim(1/(s+1), u', t)
#     return out.y[1,:]
# end

"plots frequency response of all nodes to w00"
function pltFRSNeulet(N::Int64)
    p = plot(layout=(N,N))
    w = [10.0^t for t in range(-2.0,2.0,10000)]
    for (m,n) in collect(Iterators.product(1:N,1:N))
        r = frSNeulet(N,(1,1),(m,n),w)
        plot!(w,abs.(r);
              subplot=N*(n-1)+m,
              legend=false,
              # xlims=(1e-2,1e0),
              # ylims=(-1e-2,0.25),
              xscale=:log10)
        display(p)
    end
    return p
end

# "from noise on center row to south point"
# function simNeuletFaux1D(N::Int64, tend::Float64=50.0)
#     if iseven(N)
#         error("pick an odd number svp")
#     end
#     nt = 2^23
#     nt2 = div(nt,2)
#     tpad = 1000*tend
#     dt = tpad/nt
#     dw = 2π/tpad
#     t = dt*(0:nt-1)
#     w = dw*(-nt2:nt2-1)
#     v00 = zeros(nt)
#     u = vcat(randn(nt2), zeros(nt2))
#     uw = fft(u)
#     k = cld(N+1,2)
#     for l = 1:N
#         vhat = frSNeulet(N,(k,l),(N,k),w)
#         v = real(ifft(fftshift(vhat).*uw))
#         v00 += v
#     end
#     it = floor(Int, tend/dt)
#     return t[1:it], v00[1:it]
# end

# "eigs of Laplacian with mixed Dirichlet Neumann BCs"
# function eggNeulet(N::Int64)
#     θ = [(2k+1)π/(2N+1) for k in 0:N-1]
#     eggs = 4sin.(θ/2).^2
#     eggvec = 2/sqrt(2N+1)*reshape(vcat([sin.(j*θ) for j in 1:N]...),(N,N))'
#     return eggs, eggvec
# end

"Laplacian with mixed Dirichlet Neumann BCs with my sign convention"
function lapNeulet(N::Int64)
    d = fill(-2.0,N)
    d[end] = -1.0
    dl = fill(1.0,N-1)
    L = Tridiagonal(dl, d, dl)
end

# "W in frequency domain T'WT"
# function WhatN(N::Int64, k::Int64, l::Int64)
#     θ = π/(2N+1)
#     What = [sin((2m-1)*k*θ)*sin((2n-1)*l*θ) for (m,n) in collect(Iterators.product(1:N,1:N))]
#     return 4What/(2N+1)
# end

# function matSij(N::Int64)
#     θ = π/(2N+1)
#     k = 1
#     l = 1
#     S = zeros(N,N)
#     for (m,n) in collect(Iterators.product(1:N,1:N))
#         Sij = 0
#         for (i,j) in collect(Iterators.product(1:N,1:N))
#             σij = sin((2i-1)θ/2)^2+sin((2j-1)θ/2)^2
#             Sij += sin((2i-1)k*θ)*sin((2j-1)l*θ)*sin((2i-1)m*θ)*sin((2j-1)n*θ)/(1+4σij)
#         end
#         S[m,n] = Sij
#     end
#     return S
# end
