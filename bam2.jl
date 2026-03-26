using LinearAlgebra
using ControlSystems
using NumericalIntegration
using Plots

#=====FAST (FOURIER + DIRECT NORM) AND QUITE RELIABLE (THOUGH ISZERO()?)=====#
function pltvar2(Ns,numex::Int=1)
    nN = length(Ns)
    var1 = zeros(nN)
    varn = zeros(nN)
    for i in 1:nN
        N = Ns[i]
        var1[i] = varSerrFour(N,1,numex)
        varn[i] = varSerrFour(N,N,numex)
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

"calculates variance via F' S_err F"
function varSerrFour(N::Int, h::Int, numex)
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
    var = zeros(Float64, Threads.nthreads())
    Threads.@threads for (k,l) in collect(Iterators.product(1:2N+1,1:2N+1))
        ϕkl = omg^(m*(l-k))*(1-omg^(h*(k-1))) # separate the complex phase to enable norm()
        σkl = sin(π*(k-1)/(2N+1))^2+sin(π*(l-1)/(2N+1))^2
        Skl = minreal(p/(1+4g*σkl))
        if iszero(ϕkl) # avoid abs(0.0)*norm(1/s) = 0*Inf = NaN
            continue
        else
            var[Threads.threadid()] += (abs(ϕkl)*norm(Skl))^2
        end
        # omitted *2 because /sqrt(2d) anyway
    end
    return sum(var)/(2N+1)^2
end

#=====LESS SLOW (FOURIER SINGLE DOUBLE SUM BUT INTEGRATE FREQRESP) AND QUITE RELIABLE=====#
function pltSbam(Ns)
    nN = length(Ns)
    var1 = zeros(nN)
    varn = zeros(nN)
    for i in 1:nN
        N = Ns[i]
        var1[i] = varSFbam(N,1)
        varn[i] = varSFbam(N,N)
    end
    plot(Ns, var1/2; c=:steelblue) # 1/2 as in 1/sqrt(2d) where d=lattice dimension
    plot!(Ns, varn/2; legend=false)
end

"calculates variance via integrating freqresp() curves"
function varSFbam(N::Int, h::Int)
    m = N+1
    var = zeros(Float64, Threads.nthreads())
    w = [10.0^t for t in range(-4.0,4.0,10000)]
    Threads.@threads for (k,l) in collect(Iterators.product(1:2N+1,1:2N+1))
        S0 = SN2F(N,(m,m),(k,l))
        Sn0 = SN2F(N,(m-h,m),(k,l))
        # S0n = SN2F(N,(m,m-h),(k,l))
        # var += norm([Smin(S0,Sn0); Smin(S0,S0n)])^2
        # norm() can't handle complex coeffs though
        r0 = dropdims(freqresp(S0,w); dims=(1,2))
        rn0 = dropdims(freqresp(Sn0,w); dims=(1,2))
        var[Threads.threadid()] += integrate(w, (abs.(r0-rn0)).^2)*2/π
    end
    return sum(var)/(2N+1)^2
end

"(k,l) element of F'SF which removes double summation in each element of the S matrix"
function SN2F(N::Int, (m,n)::Tuple{Int,Int}, (k,l)::Tuple{Int,Int})
    p = tf(1, [1,0]) #tf(1, [1,0,0])
    c = 1 # tf([1,1],[1])
    g = p*c
    omg = exp(-im*2π/(2N+1))
    σij = sin(π*(k-1)/(2N+1))^2+sin(π*(l-1)/(2N+1))^2
    Skl = p*omg^(k*(1-m)+l*(n-1)+(m-n))/(1+4g*σij)
    return Skl
end

#=====SLOW (DOUBLE DOUBLE SUM) AND UNRELIABLE (SMIN + MINEREAL)=====#
# "variance of 2d bamieh consensus"
# function varSbam(N::Int, n::Int)
#     m = N+1
#     var = 0
#     for (k,l) in collect(Iterators.product(1:2N+1,1:2N+1))
#         S0 = SN2cs(N,(k,l),(m,m))
#         Sn0 = SN2cs(N,(k,l),(m-n,m))
#         S0n = SN2cs(N,(k,l),(m,m-n))
#         var += norm([Smin(S0,Sn0); Smin(S0,S0n)])^2
#     end
#     return var
# end

# "analytic sensitivity function for 2d lattice with cyclic laplacian"
# function SN2cs(N::Int, (k,l)::Tuple{Int,Int}, (m,n)::Tuple{Int,Int})
#     S = 0
#     p = tf(1, [1,0])
#     c = 1
#     g = p*c
#     omg = exp(-im*2π/(2N+1))
#     for (i,j) in collect(Iterators.product(0:2N,0:2N))
#         σij = sin(π*i/(2N+1))^2+sin(π*j/(2N+1))^2
#         S += p*omg^((k-m)*i+(n-l)*j)/(1+4g*σij)
#     end
#     return minereal(S/(2N+1)^2)
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
#     num = padnum(num0,den0)-padnum(numn,denn)
#     T = minreal(tf(reverse(num),reverse(den0)))
# end

# function padnum(num,den)
#     num0 = vcat(num, fill(0,length(den)-length(num)))
# end
