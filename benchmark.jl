using BenchmarkTools, Plots, GeometricMultigrid, TunedSmoother

begin
    smoothers = [Jacobi!,GS!,SOR!,pseudo!]
    suite = BenchmarkGroup()
    A,b = TunedSmoother.static(;n=64,T=Float64)
    for s ∈ smoothers
        suite[s] = @benchmarkable mg!(st;mxiter=1,smooth! = $s) setup=(st=state($A,$b))
    end
end
results = run(suite)
time_factor = [minimum(results[s]).time/minimum(results[Jacobi!]).time for s ∈ smoothers[2:end]]

begin
    itcount(d,s;kw...) = mg!(state(d...);reltol=1e-3,smooth! = s, mxiter=32, kw...)
    avecount(data,s;kw...) = sum(itcount(d,s;kw...) for d ∈ data)/length(data)
end
data = create_synthetic(;n=64)
counts = [avecount(d,s) for s in smoothers[2:end], (name,d) ∈ data]

begin # inner=2
    time_factor = [ 2.5587099273183047
                    2.6826817316441263
                    1.3066736339751297]   
    counts = [ 3.0  4.0  1.64  1.3   3.9   3.6
                4.0  3.0  1.68  1.28  3.95  3.0
                3.0  3.0  1.14  1.28  3.23  3.84]
    counts.*time_factor
end

begin
    using StatsPlots,CategoricalArrays
    cats = ["Gauss-Sidel","SOR","Ã⁻¹ union"]
    n = length(data)
    colors = repeat([palette(:default)[2],palette(:default)[4],palette(:default)[3]],inner=n)
    ctg = CategoricalArray(repeat(cats,inner=n))
    levels!(ctg,cats)
    groupedbar((counts.*time_factor)',size=(400,400),
                group=ctg, legend=:bottomright,c=colors,
                yaxis=("relative time"),
                xaxis=("cases",(1:n,keys(data)),60))
end
savefig("synthetic_timing.png")