import GeometricMultigrid: δ,inside

# set up examples
function static(;n=32,N=2,T::Type=Float32,m=rand(T,N))
    dims = ntuple(i->n+2,N)
    L,x = zeros(T,dims...,N), zeros(T,dims)
    m ./= √sum(abs2,m)
    for I ∈ inside(x)
        x[I] = sum( @. m*I.I)
        for i ∈ 1:N 
            I.I[i] !=2 && (L[I,i] = 1)
        end
    end
    A,x = Poisson(L),FieldVector(x)
    A,A*x
end

function dipole(;n=32,N=2,T::Type=Float32,x0=rand(T,N),m=rand(T,N),s=rand(),Δt=1)
    dims = ntuple(i->n+2,N)
    L,b = zeros(T,dims...,N), zeros(T,dims)
    x0 .*= n/3; x0 .+= (n/3); s *= n/4; m ./= √sum(abs2,m)
    f(x) = sum(@. m*(x-x0))*exp(-s^2*sum(abs2,x.-x0))
    for I ∈ inside(b)
        b[I] = f(I.I)
        for i ∈ 1:N 
            I.I[i] !=2 && (L[I,i] = Δt)
        end
    end
    b .-= sum(b)/n^N
    Poisson(L),FieldVector(b)
end

@fastmath kern₀(d) = 0.5+0.5d+0.5sin(π*d)/π
μ₀(d) = kern₀(clamp(d,-1,1))
function sphere(;n=32,N=2,T::Type=Float32,x0=rand(T,N),m=rand(T,N),s=rand(),Δt=1)
    dims = ntuple(i->n+2,N)
    L,b = zeros(T,dims...,N), zeros(T,dims)
    x0 .*= n/3; x0 .+= (n/3); s *= n/4; m ./= √sum(abs2,m)
    d(x) = √sum(abs2,x.-x0)-s
    for I ∈ inside(b), i ∈ 1:N
        μp = μ₀(d(I.I .+ 0.5 .* δ(i,N).I))
        μm = μ₀(d(I.I .- 0.5 .* δ(i,N).I))
        I.I[i] !=2 && (L[I,i] = Δt*μm)
        b[I] += Δt*m[i]*(μp-μm)
    end
    Poisson(L),FieldVector(b)
end

using DataStructures
function create_synthetic(;len=100,kw...)
    OrderedDict(
        "2D-static"=>[static(N=2;kw...) for i ∈ 1:len],
        "3D-static"=>[static(N=3;kw...) for i ∈ 1:len],
        "2D-dipole"=>[dipole(N=2;kw...) for i ∈ 1:len],
        "3D-dipole"=>[dipole(N=3;kw...) for i ∈ 1:len],
        "2D-sphere"=>[sphere(N=2;kw...) for i ∈ 1:len],
        "3D-sphere"=>[sphere(N=3;kw...) for i ∈ 1:len]
    )
end