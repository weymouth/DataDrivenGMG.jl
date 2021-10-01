# count average V-cycles to get reltol 1e-3
itcount(d,s;kw...) = mg!(state(d...);reltol=1e-3,smooth! = s, mxiter=32, kw...)
avecount(data,s;kw...) = sum(itcount(d,s;kw...) for d ∈ data)/length(data)

# measure average residual reduction after it V-cycles
Δresid(st;it=1,kw...) = (resid = mg!(st;mxiter=it,log=true,kw...);
                        log10(resid[end]/resid[1])/it)
loss(data;p=p₀,kw...) = sum(Δresid(state(d...;p);kw...) for d ∈ data)/length(data)

# optimize parameters to minimize loss
using Optim
fit(data,p₀=zeros(Float32,5);smooth! = pseudo!,kw...) = Optim.minimizer(
    optimize(p->loss(data;p,smooth!,kw...),p₀, Newton(),
    Optim.Options(time_limit=60,show_trace=true); autodiff = :forward))