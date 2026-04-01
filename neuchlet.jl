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
    # v00 = zeros(nt)
    m = div(N+1,2)
    # for (m,n) in collect(Iterators.product(1:N,1:N))
        vhat = frSNeulet(N,(m,m),(m,m),w)
        uu = vcat(sin.(0.1t[1:nt2]), zeros(nt2))
        uw = fft(uu) # fft(vcat(randl(t[1:nt2]), zeros(nt2)))
        v = real(ifft(fftshift(vhat).*uw))
        # v00 += v
    # end
    plot(t[1:it],v[1:it]; c=:steelblue, label=false)
    # return t[1:it], v00[1:it]
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
# "random noise passed through an LPF"
# function randl(t)
#     nt = length(t)
#     u = randn(nt)
#     s = tf("s")
#     out = lsim(1/(s+1), u', t)
#     return out.y[1,:]
# end

"a bunch of freq response curves for varying N"
function pltFRSNeulet(Ns::Vector{Int64})
    p = plot()
    w = [10.0^t for t in range(-2.0,2.0,10000)]
    for N in Ns
        k = div(N+1,2)
        vhat = frSNeulet(N,(k,k),(k,k),w)
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

# "plots frequency response of all nodes to w00"
# function pltFRSNeuletkk(N::Int64)
#     p = plot()
#     w = [10.0^t for t in range(-2.0,2.0,10000)]
#     k = div(N+1,2)
#     for l = 1:k
#         r = frSNeulet(N,(l,l),(k,k),w)
#         plot!(w,abs.(r);
#               color=:steelblue,
#               # subplot=N*(n-1)+m,
#               legend=false,
#               # xlims=(1e-2,1e0),
#               ylims=(1e-10,2),
#               xscale=:log10,
#               yscale=:log10)
#         display(p)
#     end
#     return p
# end

# "eigs of Laplacian with mixed Dirichlet Neumann BCs"
# function eggNeulet(N::Int64)
#     θ = [(2k+1)π/(2N+1) for k in 0:N-1]
#     eggs = 4sin.(θ/2).^2
#     eggvec = 2/sqrt(2N+1)*reshape(vcat([sin.(j*θ) for j in 1:N]...),(N,N))'
#     return eggs, eggvec
# end

# "Laplacian with mixed Dirichlet Neumann BCs with my sign convention"
# function lapNeulet(N::Int64)
#     d = fill(-2.0,N)
#     d[end] = -1.0
#     dl = fill(1.0,N-1)
#     L = Tridiagonal(dl, d, dl)
# end

# "W in frequency domain T'WT"
# function WhatN(N::Int64, k::Int64, l::Int64)
#     θ = π/(2N+1)
#     What = [sin((2m-1)*k*θ)*sin((2n-1)*l*θ) for (m,n) in collect(Iterators.product(1:N,1:N))]
#     return 4What/(2N+1)
# end
