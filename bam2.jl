using LinearAlgebra
# using ControlSystems
# using NumericalIntegration
using Plots

#=====FAST (FOURIER + ANALYTICAL NORM) AND QUITE RELIABLE =====#
function pltvar(Ns,numex::Int=1)
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
    omg = exp(-im*2π/(2N+1))
    var = zeros(Float64, Threads.nthreads())
    Threads.@threads for (k,l) in collect(Iterators.product(1:2N+1,1:2N+1))
        ϕkl = 1-omg^(h*(k-1)) # separate the complex phase to enable norm()
        σkl = sin(π*(k-1)/(2N+1))^2+sin(π*(l-1)/(2N+1))^2
        # Skl = minreal(p/(1+4g*σkl))
        if iszero(ϕkl) # avoid abs(0.0)*norm(1/s) = 0*Inf = NaN
            continue
        else
            if numex == 1
                var[Threads.threadid()] += abs2(ϕkl)/(8σkl)
            elseif numex == 2
                # var[Threads.threadid()] += abs2(ϕkl)*norm(Skl)^2
                var[Threads.threadid()] += abs2(ϕkl)/(32σkl^2)
            else
                error("enter either 1 (consensus) or 2 (vehicle) svp")
            end
        end
    end
    return sum(var)/(2N+1)^2/2 # factor 1/2=2/(sqrt(2d))^2
end

#=====FAST (FOURIER + ANALYTICAL NORM) AND QUITE RELIABLE FOR 1D =====#
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
    omg = exp(-im*2π/(2N+1))
    var = 0.0
    for k = 1:2N+1
        ϕk = 1-omg^(h*(k-1)) # separate the complex phase to enable norm()
        σk = sin(π*(k-1)/(2N+1))^2
        # Sk = minreal(p/(1+4g*σk))
        if iszero(ϕk) # avoid abs(0.0)*norm(1/s) = 0*Inf = NaN
            continue
        else
            if numex == 1
                var += abs2(ϕk)/(8σk)
            elseif numex == 2
                # var += abs2(ϕk)*norm(Sk)^2
                var += abs2(ϕk)/(32σk^2)
            else
                error("enter either 1 (consensus) or 2 (vehicle) svp")
            end
        end
    end
    return var/(2N+1)/2 # factor 1/2=(1/sqrt(2d))^2
end
