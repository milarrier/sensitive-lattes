using LinearAlgebra
using ControlSystems
using FFTW
using Plots
using Random

function simF1D(N::Int64, tend::Float64=50.0)
    nt = 2^23
    nt2 = div(nt,2)
    tpad = 1024*tend
    dt = tpad/nt
    dw = 2π/tpad
    t = dt*(0:nt-1)
    w = dw*(-nt2:nt2-1)
    v00 = zeros(nt)
    # u = vcat(randn(Xoshiro(1), nt2), zeros(nt2))
    # uw = fft(u)
    for k = 1:2N+1
        vhat = frSN1D(N,k,2N+1,w)
        u = vcat(randn(nt2), zeros(nt2))
        uw = fft(u)
        v = real(ifft(fftshift(vhat).*uw))
        v00 += v
    end
    it = floor(Int, tend/dt)
    return t[1:it], v00[1:it]
end

function frSN1D(N::Int64, k::Int64, m::Int64, w)
    nw = length(w)
    r = zeros(ComplexF64, nw, Threads.nthreads())
    p = tf(1, [0.1,1,0,0])
    c = tf([2,1], [0.05,1])
    # p = tf(1, [1,1])
    # c = tf(1, [1,0])
    g = p*c
    omg = exp(-im*2π/(2N+1))
    Threads.@threads for i = 0:2N
        σi = sin(π*i/(2N+1))^2
        Si = omg^((k-m)*i)/(1+4g*σi) # w(k) -> v(m)
        r[:,Threads.threadid()] += dropdims(freqresp(Si,w); dims=(1,2))
    end
    return sum(r, dims=2)[:,1]/(2N+1)
end

#=============================== SANITY CHECKS ================================#
# "a bunch of freq response curves from w(N+1) to v(N+1) for varying N"
# function pltFRSN1D(Ns::Vector{Int64})
#     p = plot()
#     w = [10.0^t for t in range(-2.0,2.0,10000)]
#     for N in Ns
#         r = frSN1D(N,N+1,N+1,w)
#         plot!(w, abs.(r);
#               palette=palette(:Blues),
#               xlims=(1e-2,1e2),
#               # ylims=(1e-5,1e1),
#               # yscale=:log10,
#               xscale=:log10,
#               label="N="*string(N))
#         display(p)
#     end
#     return p
# end

"verifies lo-fi peaks at the same location for varying disturbance locations"
function pltFR1Dl(N::Int64)
    p = plot()
    w = [10.0^t for t in range(-2.0,2.0,10000)]
    for l = 1:N+1
        r = frSN1D(N,l,N+1,w)
        plot!(w,abs.(r);
              color=:steelblue,
              # xlims=(1e-2,1e2),
              ylims=(1e-10,2),
              # label="l="*string(l),
              legend=false,
              xscale=:log10,
              yscale=:log10)
        display(p)
    end
    return p
end

# "simulates freq->time response from w(N+l+1) to v(N+1)"
# function simF0(N::Int64)
#     p = plot!()
#     tend = 50.0
#     nt = 2^23
#     nt2 = div(nt,2)
#     tpad = 1024*tend
#     dt = tpad/nt
#     dw = 2π/tpad
#     t = dt*(0:nt-1)
#     w = dw*(-nt2:nt2-1)
#     it = floor(Int, tend/dt)
#     u = vcat(randn(Xoshiro(1),nt2), zeros(nt2))
#     uw = fft(u)
#     for l = N-2:N
#         vhat = frSN1D(N,l,N+1,w)
#         v = real(ifft(fftshift(vhat).*uw))
#         plot!(t[1:it],v[1:it];
#               subplot=2,#l-N+2,
#               label="l="*string(l))
#         display(p)
#     end
#     return p
# end
