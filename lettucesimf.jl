using LinearAlgebra
using ControlSystems
using FFTW
using Plots

function pltSim(N::Int64,tend::Float64=5.0)
    tw, vw = simF(N)
    p = plot(tw, vw; ylims=(-2,2))
    dt = 0.01;
    t = 0:dt:tend;
    u = ufn(t)
    v,tout,x,uout = lsim(minreal(ss(SN(N))), u', t) # maybe use state-space model in SN()?
    plot!(t, v[1,:], line=:dot);
    return p
end

ufn(t) = sin.(10t)

function simF(N::Int64,tend::Float64=5.0)
    # tend = 5
    nt = 2^23
    nt2 = div(nt,2)
    tpad = 1000*tend
    dt = tpad/nt
    dw = 2π/tpad
    t = dt*(0:nt-1)
    w = dw*(-nt2:nt2-1) # w = (-π/dt):dw:(π/dt-dw) # length(w) != length(t) ???
    vhat = frSN(N,w)
    u = vcat(ufn(t[1:nt2]), zeros(nt2))
    uf = fft(u)
    v = real(ifft(fftshift(vhat).*uf))
    it = floor(Int, tend/dt)
    # plot(t[1:it],v[1:it]) # plot rendering is so slow for sizes like 2^23
    return t[1:it], v[1:it]
end

function pltFR(Ns)
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

"frequency response of SN calculated node-wise"
function frSN(N::Int64,w)
    nw = length(w)
    iw0 = findall(iszero, w)
    g = openloop()
    r = zeros(nw) # need to handle zero frequency separately
    for j = 0:2N
        for k = 0:2N
            Sjk = 1/(1+4g-2g*(cos(2π*j/(2N+1))+cos(2π*k/(2N+1))))
            r = r + dropdims(freqresp(Sjk,w); dims=(1,2))
        end
    end
    if !isempty(iw0)
        r[iw0[1]] = 1
    end
    r = r/(2N+1)^2
end

function SN(N::Int64)
    g = openloop()
    S = 0
    for j = 0:2N
        for k = 0:2N
            S += 1/(1+4g-2g*(cos(2π*j/(2N+1))+cos(2π*k/(2N+1))))
        end
    end
    S = S/(2N+1)^2
    return minreal(S)
end

function openloop()
    p = tf(1, [0.1,1,0,0])
    c = tf([2,1], [0.05,1])
    # p = tf(1, [1,1])
    # c = tf(1, [1,0])
    g = p*c
end

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
