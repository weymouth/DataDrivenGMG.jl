using GeometricMultigrid: mg!,Vcycle!,norm
using Optim

# count average V-cycles to get reltol 1e-3
itcount(d,s;kw...) = mg!(state(d...;smooth! = s,kw...);reltol=1e-3,abstol=0,smooth! = s, mxiter=32, kw...)
avecount(data,s;kw...) = sum(itcount(d,s;kw...) for d ∈ data)/length(data)

# measure average residual reduction after it V-cycles
function Δresid(st;it=1,kw...)
    r₀ = norm(st.r)
    for _ in 1:it; Vcycle!(st;kw...); end
    return log10(norm(st.r)/r₀/it)
end
loss(data;p=p₀,kw...) = sum(Δresid(state(d...;p,kw...);p,kw...) for d ∈ data)/length(data)

# optimize parameters to minimize loss
fit(data,p₀=zeros(Float32,5);smooth! = pseudo!,kw...) = Optim.minimizer(
    optimize(p->loss(data;p,smooth!,kw...),p₀, Newton(),
    Optim.Options(time_limit=120,show_trace=true,f_tol=1e-4); autodiff = :forward))
