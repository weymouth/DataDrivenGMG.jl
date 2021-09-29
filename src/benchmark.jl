using LinearAlgebra, GeometricMultigrid, BenchmarkTools, Plots

begin
    include("candidate_smoothers.jl")
    smoothers = [Jacobi!,GS!,SOR!,pseudo!]
end
begin
    include("create_synthetic.jl")
    suite = BenchmarkGroup()
    A,b = static(;n=128,T=Float32)
    for s ∈ smoothers
        suite[s] = @benchmarkable mg!(st;mxiter=1,inner=2,smooth! = $s) setup=(st=mg_state($A,zero($b),$b,pseudo=true))
    end
end
results = run(suite)
time_factor = [minimum(results[s]).time/minimum(results[Jacobi!]).time for s ∈ [2:end]]
begin
    time_factor = [2.01557, 2.61135, 1.36815]
end

begin
    include("create_synthetic.jl")
    data = create_synthetic()
end
begin
    itcount(A,b,s) = mg!(mg_state(A,zero(b),b,pseudo=true);reltol=1e-3,inner=2,smooth! = s,mxiter=32)
    avecount(data,s) = sum(itcount(d...,s) for d ∈ data)/length(data)
    counts = [avecount(d,s) for s in smoothers[2:end], (name,d) ∈ data]
end
begin
    counts = [ 3.0  3.0  2.64  2.33  3.65  3.08
                3.0  4.0  2.53  2.5   3.05  3.92
                3.0  3.0  1.88  2.27  3.17  3.0]
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
savefig("synthetic_scaling.png")