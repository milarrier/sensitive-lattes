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

"Bamieh metric scaling behavior"
function pltvar1(Ns,numex::Int=1)
    nN = length(Ns)
    var1 = zeros(nN)
    varn = zeros(nN)
    for i in 1:nN
        N = Ns[i]
        var1[i] = var1SerrFour(N,1,numex)
        varn[i] = var1SerrFour(N,N,numex)
    end
    if numex == 1
        p = plot(Ns, var1; c=:steelblue)
        plot!(Ns, varn; legend=false)
    else
        p = plot(Ns, var1; c=:steelblue, layout=(2,1), subplot=1, legend=false)
        plot!(Ns, varn; c=:steelblue, subplot=2, legend=false)
    end
    display(p)
    return var1,varn
end

"calculates variance via F' S_err"
function var1SerrFour(N::Int, h::Int, numex)
    if numex == 1 # consensus
        p = tf(1, [1,0]) # 1/s
        c = 1
        g = p*c
    elseif numex == 2 # vehicular formation
        p = tf(1, [1,0,0]) # 1/s^2
        c = tf([1,1],[1]) # 1+s
        g = p*c
    else
        error("enter either 1 (consensus) or 2 (vehicle) svp")
    end
    m = N+1
    omg = exp(-im*2π/(2N+1))
    var = 0.0
    for k = 1:2N+1
        ϕk = omg^((1-m)*(k-1))*(1-omg^(h*(k-1))) # separate the complex phase to enable norm()
        σk = sin(π*(k-1)/(2N+1))^2
        Sk = minreal(p/(1+4g*σk))
        if iszero(ϕk) # avoid abs(0.0)*norm(1/s) = 0*Inf = NaN
            continue
        else
            var += (abs(ϕk)*norm(Sk))^2
        end
    end
    return var/(2N+1)/2 # factor 1/2=(1/sqrt(2d))^2
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
