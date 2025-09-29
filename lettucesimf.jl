using LinearAlgebra
using ControlSystems
using FFTW
using Plots

# function simF(N)
#     T = 5
#     nt = 2^23
#     Tpad = 1000*T
#     dt=Tpad/nt
#     dw=2π/Tpad
#     t = dt*(0:nt-1)
#     w = -π/dt:dw:(π/dt-dw)
#     vhat = frSN(N,w)
#     u = [sin.(10t[1:nt/2]); zeros(nt/2,1)]
#     Uw = fft(u)
#     v = real(ifft(fftshift(vhat).*Uw))
# end

function pltFR(Ns)
    p = plot()
    w = [10.0^t for t in range(-2.0,2.0,10000)]
    for N in Ns
        vhat = frSN(N,w)
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

"frequency response of SN"
function frSN(N,w)
    nw = length(w)
    p = tf(1, [0.1,1,0,0])
    c = tf([2,1], [0.05,1])
    # p = tf(1, [1,1])
    # c = tf(1, [1,0])
    g = p*c
    r = zeros(nw)
    for j = 0:2N
        for k = 0:2N
            Sjk = 1/(1+4g-2g*(cos(2π*j/(2N+1))+cos(2π*k/(2N+1))))
            r = r + dropdims(freqresp(Sjk,w); dims=(1,2))
        end
    end
    r = r/(2N+1)^2
end

function pltFrSjk(N)
    h = plot()
    w = [10.0^t for t in range(-2.0,2.0,10000)]
    p = tf(1, [1,1])
    c = tf(1, [1,0])
    g = p*c
    for j = 0:2N
        for k=2
            Sjk = 1/(1+4g-2g*(cos(2π*j/(2N+1))+cos(2π*k/(2N+1))))
            r = dropdims(freqresp(Sjk,w); dims=(1,2))
            plot!(w, abs.(r);
                  palette=reverse(cgrad(:Blues)),
                  xlims=(1e-2,1e2),
                  yscale=:log10,
                  xscale=:log10,
                  label="jk="*string(j,k))
            display(h)
        end
    end
end
