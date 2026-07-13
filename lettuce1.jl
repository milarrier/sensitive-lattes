using LinearAlgebra
using ControlSystems
# using FFTW
# using ToeplitzMatrices
using Plots

Sinf(g) = 1 / sqrt(4g+1)

function plotv0inf()
    gre = -5000:0.3:3
    gim = -200:0.3:3
    v = [Sinf(x + im*y) for y in gim, x in gre]
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

#=Warning: High-order transfer functions are highly sensitive to numerical errors=#
# function pltSbam(Ns)
#     nN = length(Ns)
#     var1 = zeros(nN)
#     varn = zeros(nN)
#     for i in 1:nN
#         var1[i] = varSbam(Ns[i])
#         varn[i] = varSbam(Ns[i],Ns[i])
#     end
#     plot(Ns, var1; c=:steelblue)
#     plot!(Ns, varn)
# end

# "variance of 1d bamieh consensus"
# function varSbam(N::Int, n::Int=1)
#     m = N+1
#     var = 0
#     for k = 1:2N+1
#         S0 = SN1cy(N,k,m)
#         Sn = SN1cy(N,k,m-n)
#         var += norm(minreal(Smin(S0,Sn)))^2
#     end
#     return var
# end

# "analytic sensitivity function for 1d lattice with cyclic laplacian"
# function SN1cy(N::Int,k::Int,m::Int)
#     S = 0
#     p = tf(1, [1,0])
#     c = 1
#     g = p*c
#     omg = exp(-im*2π/(2N+1))
#     for i = 0:2N
#         σi = sin(π*i/(2N+1))^2
#         S += p*omg^((k-m)*i)/(1+4g*σi) # w(k) -> v(m)
#     end
#     return minereal(S/(2N+1))
# end

# "mine minreal cleans up tiny imaginary parts"
# function minereal(S)
#     # round.(Int,...) or not doesn't seem to matter
#     num0 = real(S.matrix[1].num.coeffs)
#     den0 = real(S.matrix[1].den.coeffs)
#     T = tf(reverse(num0),reverse(den0))
# end

# function Smin(S0, Sn)
#     num0 = S0.matrix[1].num.coeffs
#     den0 = S0.matrix[1].den.coeffs
#     numn = Sn.matrix[1].num.coeffs
#     denn = Sn.matrix[1].den.coeffs
#     if den0!=denn
#         error("identical denominator assumption invalid")
#     end
#     num = padnum(num0,den0)-padnum(numn,denn);
#     T = tf(reverse(num),reverse(den0))
# end

# function padnum(num,den)
#     num0 = vcat(num, fill(0,length(den)-length(num)))
# end
