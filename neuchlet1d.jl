using LinearAlgebra
using ControlSystems
using FFTW
using Plots

"simulates in freq domain then converts to time domain"
function simNeuletF1D(N::Int64, tend::Float64=81.92)
    nt = 2^23
    nt2 = div(nt,2)
    tpad = 1024*tend
    dt = tpad/nt
    dw = 2π/tpad
    t = dt*(0:nt-1)
    w = dw*(-nt2:nt2-1)
    it = floor(Int, tend/dt)
    v00 = zeros(nt)
    # for m in 1:N
        m = div(N+1,2)
        u = vcat(randl(t[1:nt2]), zeros(nt2))
        uw = fft(u)
        vhat = frSNeulet1D(N,m,N,w)
        v = real(ifft(fftshift(vhat).*uw))
        v00 += v
    # end
    return t[1:it], v00[1:it]#, u[1:it]
end

# function randp(N,nt)
#     u = zeros(nt)
#     for i = 1:nt
#         u[i] = sol.W[i][ceil(Int,N/2)]
#     end
#     return u
# end

"random noise passed through an LPF"
function randl(t)
    nt = length(t)
    u = randn(nt)
    s = tf("s")
    out = lsim(1/(10s+1), u', t)
    return out.y[1,:]
end

"frequency response from k to m obtained node-wise and then summed up"
function frSNeulet1D(N::Int64, k::Int64, m::Int64, w)
    nw = length(w)
    r = zeros(ComplexF64, nw, Threads.nthreads())
    p = tf(1, [0.1,1,0,0])
    c = tf([2,1], [0.05,1])
    # p = tf(1, [1,1])
    # c = tf(1, [1,0])
    g = p*c
    θ = π/(2N+1)
    Threads.@threads for i in 1:N
        Si = sin((2i-1)k*θ)*sin((2i-1)m*θ)/(1+4g*sin((2i-1)θ/2)^2)
        r[:,Threads.threadid()] += dropdims(freqresp(Si,w); dims=(1,2))
    end
    r = sum(r, dims=2)
    # iw0 = findall(iszero, w) # at zero frequency it's simply zero in this case
    r = 4r/(2N+1)
end

#=============================== SANITY CHECKS ================================#
"a bunch of freq response curves for varying N"
function pltFRSNeulet1D(Ns::Vector{Int64})
    p = plot()
    w = [10.0^t for t in range(-2.0,2.0,10000)]
    for N in Ns
        k = div(N+1,2)
        vhat = frSNeulet1D(N,k,k,w)
        plot!(w, abs.(vhat);
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

"plots frequency response of all nodes to w0"
function pltFRSNeulet1kk(N::Int64)
    p = plot()
    w = [10.0^t for t in range(-2.0,2.0,10000)]
    k = div(N+1,2)
    for l = 1:k
        r = frSNeulet1D(N,l,k,w)
        plot!(w,abs.(r);
              color=:steelblue,
              legend=false,
              # xlims=(1e-2,1e0),
              ylims=(1e-10,2),
              xscale=:log10,
              yscale=:log10)
        display(p)
    end
    return p
end
