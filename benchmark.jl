using BenchmarkTools, Plots, TunedSmoother
using GeometricMultigrid: Vcycle!

data = create_synthetic(;n=64)
begin
    smoothers = [Jacobi!,GS!,SOR!,pseudo!]
    suite = BenchmarkGroup()
    for s ∈ smoothers, (name,d) ∈ data
        suite[name,s] = @benchmarkable Vcycle!(st;smooth! = $s) setup=(st=state($d[1]...))
    end
end
begin # run benchmarks and count iterations
    results = run(suite)
    times = [minimum(results[name,s]).time for s ∈ smoothers, name ∈ keys(data)]
    time_factor = times[2:end,:]./times[1,:]'
    counts = [avecount(d,s) for s in smoothers[2:end], (name,d) ∈ data]
end

begin # results from block above
    time_factor = [ 2.26743  2.29541  2.25677  2.24512  2.27092  2.30344
                    2.87374  2.46019  2.86748  2.52186  2.87211  2.44708
                    1.37096  1.56152  1.36767  1.54179  1.36671  1.54011]
    time_factor = [ 2.397728501892915
                    2.5168063818280153
                    1.3297458085451594]
    counts = [ 3.0  4.0  1.5   1.17  3.98  3.55
                4.0  3.0  1.55  1.16  3.96  3.0
                3.0  3.0  1.07  1.14  3.18  3.0]
    rel_time = counts.*time_factor
    accel = rel_time[1:2,:]./rel_time[3,:]' # speed up!!
end

begin # plot
    using StatsPlots,CategoricalArrays
    cats = ["Gauss-Sidel","SOR","Ã⁻¹ union"]
    n = length(data)
    colors = repeat([palette(:default)[2],palette(:default)[4],palette(:default)[3]],inner=n)
    ctg = CategoricalArray(repeat(cats,inner=n))
    levels!(ctg,cats)
    groupedbar((counts.*time_factor)',size=(400,350),
                group=ctg, legend=:bottomright,c=colors,
                yaxis=("relative time"),
                xaxis=("cases",(1:n,keys(data)),60))
end
savefig("synthetic_timing.png")