using LinearAlgebra
using ControlSystems
using FFTW
using Plots

function simF1D(N::Int64, tend::Float64=50.0)
    nt = 2^24
    nt2 = div(nt,2)
    tpad = 1000*tend
    dt = tpad/nt
    dw = 2π/tpad
    t = dt*(0:nt-1)
    w = dw*(-nt2:nt2-1)
    # v00 = zeros(nt)
    # for n = 1:2N+1
        vhat = frSN1D(N,2N+1,w)
        u = vcat(randn(nt2), zeros(nt2))
        uw = fft(u)
        v = real(ifft(fftshift(vhat).*uw))
        # v00 += v
    # end
    it = floor(Int, tend/dt)
    return t[1:it], v[1:it]
end

function frSN1D(N::Int64, n::Int64, w)
    nw = length(w)
    r = zeros(ComplexF64, nw, Threads.nthreads())
    p = tf(1, [0.1,1,0,0])
    c = tf([2,1], [0.05,1])
    # p = tf(1, [1,1])
    # c = tf(1, [1,0])
    g = p*c
    omg = exp(-im*2π/(2N+1))
    Threads.@threads for j = 0:2N
        σj = sin(π*j/(2N+1))^2
        Sj = omg^((N-n+1)*j)/(1+4g*σj) # w(0) => v(n)
        r[:,Threads.threadid()] += dropdims(freqresp(Sj,w); dims=(1,2))
    end
    r = sum(r, dims=2)
    # iw0 = findall(iszero, w) # need to handle zero frequency separately with this g
    # if !isempty(iw0)
    #     r[iw0[1]] = 1 # for s=0 only S00=1/1 survives the rest are 0/(0+c)
    # end
    r = r/(2N+1)
end
