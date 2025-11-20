using LinearAlgebra
using ControlSystems
# using FFTW
# using ToeplitzMatrices
using Plots

v0inf(g) = 1 / sqrt(4g+1)

function plotv0inf()
    gre = -5000:0.3:3
    gim = -200:0.3:3
    v = [v0inf(x + im*y) for y in gim, x in gre]
    contour(gre, gim, log.(abs.(v));
            c=reverse(cgrad(:ice)),
            # aspect_ratio=:equal,
            grid=false)
    xlims!(minimum(gre), maximum(gre))
    ylims!(minimum(gim), maximum(gim))
    vline!([-0.25]; ls=:dot, c=:black, lw=0.5, label="")
    hline!([0]; ls=:dot, c=:black, lw=0.5, label="")
end

function plotnyq()
    p = tf(1, [0.1,1,0,0])
    c = tf([2,1], [0.05,1])
    g = p*c
    nyre, nyim, wout = nyquist(g)
    plot!(nyre[1,1,:], nyim[1,1,:]; c=:goldenrod, label="")
end
