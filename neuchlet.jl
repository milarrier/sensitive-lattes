using LinearAlgebra

function eggNeulet(N::Int64)
    θ = [(2k+1)π/(2N+1) for k in 0:N-1]
    eggs = 4sin.(θ/2).^2
    eggvec = 2/sqrt(2N+1)*reshape(vcat([sin.(j*θ) for j in 1:N]...),(N,N))'
    return eggs, eggvec
end

function lapNeulet(N::Int64)
    d = fill(-2.0,N)
    d[end] = -1.0
    dl = fill(1.0,N-1)
    L = Tridiagonal(dl, d, dl)
end

function WhatN(N::Int64, k::Int64, l::Int64)
    θ = π/(2N+1)
    What = [sin((2m-1)*k*θ)*sin((2n-1)*l*θ) for (m,n) in collect(Iterators.product(1:N,1:N))]
    return 4What/(2N+1)
end
