using LinearAlgebra
using ControlSystems
using FFTW
using NumericalIntegration
using Plots

"simulates in freq domain then converts to time domain"
function simF(N::Int64, tend::Float64=50.0)
    nt = 2^23
    nt2 = div(nt,2)
    tpad = 1024*tend
    dt = tpad/nt
    dw = 2π/tpad
    t = dt*(0:nt-1)
    w = dw*(-nt2:nt2-1) # w = (-π/dt):dw:(π/dt-dw) # length(w) != length(t) ???
    v00 = zeros(nt)
    u = vcat(sin.(0.1t[1:nt2]), zeros(nt2))
    uw = fft(u)
    (k,l) = (N+1,N+1)
    # for (k,l) in collect(Iterators.product(1:2N+1,1:2N+1))
        vhat = frSN(N,(k,l),(N+1,N+1),w)
        # u = vcat(randn(nt2), zeros(nt2))
        # uw = fft(u)
        v = real(ifft(fftshift(vhat).*uw))
        v00 += v
    # end
    it = floor(Int, tend/dt)
    # plot(t[1:it],v[1:it]) # plot rendering is so slow for the size of 2^23
    return t[1:it], v00[1:it] #, u[1:it]
end

"frequency response of SN from (k,l) to (m,n) obtained node-wise and then summed up"
function frSN(N::Int64, (k,l)::Tuple{Int64,Int64}, (m,n)::Tuple{Int64,Int64}, w)
    nw = length(w)
    r = zeros(ComplexF64, nw, Threads.nthreads())
    p = tf(1, [0.1,1,0,0])
    c = tf([2,1], [0.05,1])
    # p = tf(1, [1,1])
    # c = tf(1, [1,0])
    # p = tf(1, [1,0])
    # c = 1
    g = p*c
    omg = exp(-im*2π/(2N+1))
    Threads.@threads for (i,j) in collect(Iterators.product(0:2N,0:2N))
        σij = sin(π*i/(2N+1))^2+sin(π*j/(2N+1))^2
        Sij = omg^(-(m-k)*i+(n-l)*j)/(1+4g*σij) # *p for input disturbance
        r[:,Threads.threadid()] += dropdims(freqresp(Sij,w); dims=(1,2))
    end
    r = dropdims(sum(r, dims=2), dims=2)
    iw0 = findall(iszero, w) # need to handle zero frequency separately with this g
    if !isempty(iw0)
        r[iw0[1]] = 1 # for s=0 only S00=1/1 survives; the rest are 0/(0+c)
    end
    r = r/(2N+1)^2
end

#=============================== SANITY CHECKS ================================#
# "compares simF() result with time domain simulation for small N"
# function pltSim(N::Int64, tend::Float64=5.0)
#     tw, vw = simF(N)
#     p = plot(tw, vw)
#     dt = 0.01;
#     t = 0:dt:tend;
#     u = ufn(t)
#     v,tout,x,uout = lsim(SN(N), u', t)
#     plot!(t, v[1,:], line=:dot);
#     return p
# end

# ufn(t) = sin.(0.1t)

# "finite sensitivity function in state-space form for lsim accuracy"
# function SN(N::Int64)
#     A = [0 1 0; 0 0 1; 0 0 -10]
#     B = [0; 0; 10]
#     C = [1 0 0]
#     Ak = -20
#     Bk = 1
#     Ck = -780
#     Dk = 40
#     p = ss(A,B,C,0.0)
#     c = ss(Ak,Bk,Ck,Dk)
#     g = minreal(p*c)
#     S = 0
#     for j = 0:2N
#         for k = 0:2N
#             S += 1/(1+4g-2g*(cos(2π*j/(2N+1))+cos(2π*k/(2N+1))))
#         end
#     end
#     return minreal(S)/(2N+1)^2
# end

"a bunch of freq response curves for varying N"
function pltFR(Ns::Vector{Int64})
    p = plot()
    w = [10.0^t for t in range(-2.0,2.0,10000)]
    for N in Ns
        r = frSN(N,(N+1,N+1), (N+1,N+1), w)
        plot!(w, abs.(r);
              palette=:Blues,
              xlims=(1e-2,1e2),
              # ylims=(1e-5,1e1),
              yscale=:log10,
              xscale=:log10,
              label="N="*string(N))
        display(p)
    end
    return p
end

# "plots frequency response of all nodes to w00"
# function pltFRSN(N::Int64)
#     p = plot()
#     w = [10.0^t for t in range(-2.0,2.0,10000)]
#     for k = 1:N+1
#         for l = k
#             r = frSN(N,(k,l),(N+1,N+1),w)
#             plot!(w,abs.(r);
#                   color=:steelblue,
#                   # xlims=(1e-2,1e0),
#                   ylims=(1e-6,2),
#                   xscale=:log10,
#                   yscale=:log10,
#                   legend=false)#:bottomleft,
#                   # label=string((k,l)))
#             display(p)
#         end
#     end
#     return p
# end

"numerical integration to get h2 norm of bamieh consensus"
function pltSbam(Ns)
    nN = length(Ns)
    varel = zeros(nN)
    varbs = zeros(nN)
    for i in 1:nN
        N = Ns[i]
        varel[i] = varSbam(N,1)
        varbs[i] = varSbam(N,N)
    end
    plot(Ns, varel; c=:steelblue)
    plot!(Ns, varbs; legend=false)
end

function varSbam(N::Int, n::Int)
    m = N+1
    var = 0
    w = [10.0^t for t in range(-4.0,4.0,10000)]
    for (k,l) in collect(Iterators.product(1:2N+1,1:2N+1))
        r0 = frSN(N,(k,l),(m,m),w)
        # why does the simplification produce smaller values?
        # r1 = frSN(N,(k,l),(m-n,m),w)
        # var += sqrt(integrate(w, (abs.(r0-r1)).^2)*2/π)
        r10 = frSN(N,(k,l),(m-n,m),w)
        r01 = frSN(N,(k,l),(m,m-n),w)
        var += integrate(w, (abs.(r0-r10)).^2+(abs.(r0-r01)).^2)/π
    end
    return var
end
