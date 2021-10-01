using BenchmarkTools, Plots, TunedSmoother

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

data = create_synthetic(;n=64)
counts = [avecount(d,s) for s in smoothers[2:end], (name,d) ∈ data]

jacobicount = [itcount(d[1],Jacobi!,reltol=1) for (name,d) ∈ data]

begin # inner=2
    time_factor = [ 2.397728501892915
                    2.5168063818280153
                    1.3297458085451594]   
    counts = [ 3.0  4.0  1.5   1.17  3.88  3.55
                4.0  3.0  1.55  1.16  3.96  3.0
                3.0  3.0  1.07  1.14  3.18  3.0]
    rel_time = counts.*time_factor
    accel = rel_time[1:2,:]./rel_time[3,:]' # speed up!!
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