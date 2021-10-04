using BenchmarkTools, Plots, TunedSmoother
using GeometricMultigrid: Vcycle!

data = create_synthetic(;n=64)
begin
    smoothers = [Jacobi!,GS!,SOR!,pseudo!]
    suite = BenchmarkGroup()
    for s ∈ smoothers, (name,d) ∈ data
        suite[name,s] = @benchmarkable Vcycle!(st;smooth! = $s) setup=(st=state($d[1]...;smooth! = $s))
    end
end
begin # run benchmarks and count iterations
    results = run(suite)
    times = [minimum(results[name,s]).time for s ∈ smoothers, name ∈ keys(data)]
    time_factor = times[2:end,:]./times[1,:]'
end

p = Float32[-0.14102, -0.0149183, 0.00704117, 0.39624, -0.242721, 1.41382, 1.70377, 0.558954, 0.0511064]
counts = [avecount(d,pseudo!;p) for (_,d) ∈ data]

begin # results from block above
    time_factor = [ 3.50716   2.72302   3.5       2.73946   3.51437   2.71142
                    4.75072   3.24036   4.74      3.21826   4.76437   3.19066
                    1.37096   1.56152   1.36767   1.54179   1.36671   1.54011]
    counts = [ 3.0  4.0  1.5   1.17  3.98  3.55
                4.0  3.0  1.55  1.16  3.96  3.0
                2.0  3.0  1.04  1.04  3.18  3.0]
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
                group=ctg,c=colors,
                yaxis=("relative time"),
                xaxis=("cases",(1:n,keys(data)),60))
end
savefig("synthetic_timing.png")