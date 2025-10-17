using LinearAlgebra
using ControlSystems
using FFTW
using Plots

"simulates in freq domain then converts to time domain"
function simF(N::Int64, tend::Float64=50.0)
    nt = 2^23
    nt2 = div(nt,2)
    tpad = 1000*tend
    dt = tpad/nt
    dw = 2π/tpad
    t = dt*(0:nt-1)
    w = dw*(-nt2:nt2-1) # w = (-π/dt):dw:(π/dt-dw) # length(w) != length(t) ???
    v00 = zeros(nt)
    for (m,n) in collect(Iterators.product(-N:N,-N:N))
        vhat = frSN(N,m,n,w)
        u = vcat(randn(nt2), zeros(nt2)) # vcat(ufn(t[1:nt2]), zeros(nt2))
        uw = fft(u)
        v = real(ifft(fftshift(vhat).*uw))
        v00 += v
    end
    it = floor(Int, tend/dt)
    # plot(t[1:it],v[1:it]) # plot rendering is so slow for the size of 2^23
    return t[1:it], v00[1:it]
end

"frequency response of SN calculated node-wise and then summed up"
function frSN(N::Int64, m::Int64, n::Int64, w)
    nw = length(w)
    r = zeros(ComplexF64, nw, Threads.nthreads())
    p = tf(1, [0.1,1,0,0])
    c = tf([2,1], [0.05,1])
    # p = tf(1, [1,1])
    # c = tf(1, [1,0])
    g = p*c
    omg = exp(-im*2π/(2N+1))
    Threads.@threads for (j,k) in collect(Iterators.product(0:2N,0:2N))
        σjk = sin(2π*j/(2N+1))^2+sin(2π*k/(2N+1))^2
        Sjk = omg^(-m*j+n*k)/(1+4g*σjk)
        r[:,Threads.threadid()] += dropdims(freqresp(Sjk,w); dims=(1,2))
        # @show (j,k)
    end
    r = sum(r, dims=2)
    iw0 = findall(iszero, w) # need to handle zero frequency separately with this g
    if !isempty(iw0)
        r[iw0[1]] = 1 # for s=0 only S00=1/1 survives the rest are 0/(0+c)
    end
    r = r/(2N+1)^2 *omg^(m-n) # from (-m,-n) to (0,0) not sure it matters
end

#=============================== SANITY CHECKS ================================#
"compares simF() result with time domain simulation for small N"
function pltSim(N::Int64, tend::Float64=5.0)
    tw, vw = simF(N)
    p = plot(tw, vw)
    dt = 0.01;
    t = 0:dt:tend;
    u = ufn(t)
    v,tout,x,uout = lsim(SN(N), u', t)
    plot!(t, v[1,:], line=:dot);
    return p
end

# ufn(t) = sin.(10t)
# ufn(t) = sin.(t)

"finite sensitivity function in state-space form for lsim accuracy"
function SN(N::Int64)
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
    S = 0
    for j = 0:2N
        for k = 0:2N
            S += 1/(1+4g-2g*(cos(2π*j/(2N+1))+cos(2π*k/(2N+1))))
        end
    end
    return minreal(S)/(2N+1)^2
end

"a bunch of freq response curves for varying N on the same plot"
function pltFR(Ns::Vector{Int64})
    p = plot()
    w = [10.0^t for t in range(-2.0,2.0,10000)]
    for N in Ns
        vhat = frSN(N,w)
        plot!(w, abs.(vhat);
              palette=palette(:Blues, rev=true),
              xlims=(1e-2,1e2),
              # ylims=(1e-5,1e1),
              yscale=:log10,
              xscale=:log10,
              label="N="*string(N))
        display(p)
    end
    return p
end

"plots all node responses from w00"
function pltSimF00(N::Int64)
    tend = 200.0
    p = plot(layout=(2N+1,2N+1))
    for m = -N:N
        for n = -N:N
            t,v = simF(N,m,n,tend) # for a version of simF that doesn't use the internal m,n loop
            plot!(t,v, subplot=(N+m)*(2N+1)+N+n+1, legend=false)
            display(p)
        end
    end
    return p
end

"plots edge nodes far from origin"
function pltSimF(N::Int64)
    tend = 200.0
    p = plot()
    for n in [N,N-2,N-4]
        t,v = simF(N,0,n,tend)
        plot!(t,v;
              palette=palette(:Blues_7, rev=true),
              label="n="*string(n))
        display(p)
    end
    return p
end

# "threaded simF() doesn't help much and FFTW in loop seems to create lock conflicts"
# function simFth(N::Int64, tend::Float64=50.0)
#     nt = 2^23
#     nt2 = div(nt,2)
#     tpad = 1000*tend
#     dt = tpad/nt
#     dw = 2π/tpad
#     t = dt*(0:nt-1)
#     w = dw*(-nt2:nt2-1)
#     it = floor(Int, tend/dt)
#     v00 = zeros(it, Threads.nthreads())
#     Threads.@threads for (m,n) in collect(Iterators.product(-N:N,-N:N))
#         vhat = frSN(N,m,n,w)
#         u = vcat(randn(nt2), zeros(nt2))
#         v = real(vhat.*u)
#         uw = fft(u)
#         v = real(ifft(fftshift(vhat).*uw))
#         v00[:,Threads.threadid()] += v[1:it]
#     end
#     v00 = sum(v00, dims=2)
#     return t[1:it], v00
# end

# function frSN1(N::Int64, m::Int64, n::Int64, w)
#     nw = length(w)
#     r = zeros(nw)
#     p = tf(1, [0.1,1,0,0])
#     c = tf([2,1], [0.05,1])
#     # p = tf(1, [1,1])
#     # c = tf(1, [1,0])
#     g = p*c
#     omg = exp(-im*2π/(2N+1))
#     for (j,k) in collect(Iterators.product(0:2N,0:2N))
#         σjk = sin(2π*j/(2N+1))^2+sin(2π*k/(2N+1))^2
#         Sjk = omg^(-m*j+n*k)/(1+4g*σjk)
#         r = r + dropdims(freqresp(Sjk,w); dims=(1,2))
#         # @show (j,k)
#     end
#     iw0 = findall(iszero, w) # need to handle zero frequency separately
#     if !isempty(iw0)
#         r[iw0[1]] = 1 # for s=0 only S00=1/1 survives the rest are 0/(0+c)
#     end
#     r = r/(2N+1)^2 *omg^(m-n) # from (-m,-n) to (0,0) not sure it matters
# end

# function pltFrSjk(N)
#     h = plot()
#     w = [10.0^t for t in range(-2.0,2.0,10000)]
#     p = tf(1, [1,1])
#     c = tf(1, [1,0])
#     g = p*c
#     for j = 0:2N
#         for k=2
#             Sjk = 1/(1+4g-2g*(cos(2π*j/(2N+1))+cos(2π*k/(2N+1))))
#             r = dropdims(freqresp(Sjk,w); dims=(1,2))
#             plot!(w, abs.(r);
#                   palette=reverse(cgrad(:Blues)),
#                   xlims=(1e-2,1e2),
#                   yscale=:log10,
#                   xscale=:log10,
#                   label="jk="*string(j,k))
#             display(h)
#         end
#     end
# end
