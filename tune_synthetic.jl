using TunedSmoother, Plots, DataStructures
using GeometricMultigrid: mg!
using TunedSmoother: p₀

# create synthetic example systems
begin
    data = create_synthetic(len=25)
    opt = OrderedDict(name=>p₀ for name ∈ keys(data))
    opt["union"] = p₀
end

# optimize Parameterized smoother
begin
    for (name,train) ∈ data
        @show name
        opt[name] = fit(train,opt[name])
    end
    @show "union"
    opt["union"] = fit(vcat((d for (_,d) in data)...),opt["union"])
    opt
end
begin # more training?
    name = "3D-static"
    opt[name] = fit(data[name],opt[name])        
end
begin
    opt = OrderedDict(
        "2D-static" => [-2.26078, -1.35812, -0.198977, 0.467383, -0.260235, 0.0687914, 0.0520279, -0.000723798, 0.00639344],
        "3D-static" => [0.806427, 0.1904, 0.0066247, 0.554321, -0.260235, -0.650988, -0.241448, -0.0228976, -0.00349145],
        "2D-dipole" => [2.80018, 1.71771, 0.25784, 0.419634, -0.260235, 0.803588, 0.530899, 0.00125979, -0.0214498],
        "3D-dipole" => [0.115248, -0.000663957, 0.000381931, 0.389719, -0.260235, 0.0322515, 0.101498, 0.00717328, -0.00287373],
        "2D-sphere" => [0.0915379, 0.0833613, 0.017635, 0.447855, -0.291, 0.687612, 1.31916, 0.695357, 0.0940616],
        "3D-sphere" => [-0.00827322, -0.0546565, -0.00438196, 0.53397, -0.377591, 0.446063, -0.0336524, -0.0570324, -0.0110678],
        "union"     => [-0.14102, -0.0149183, 0.00704117, 0.39624, -0.242721, 1.41382, 1.70377, 0.558954, 0.0511064])
    # opt = OrderedDict(
    #     "2D-static" => Float32[-2.46273, -1.32142, -0.175033, 0.145184, 0],
    #     "3D-static" => Float32[0.93542, 0.210713, 0.00503395, 0.309537, 0.0135281],
    #     "2D-dipole" => Float32[-2.56297, -1.41248, -0.188902, 0.163554, 0],
    #     "3D-dipole" => Float32[-0.752343, -0.258171, -0.0168188, 0.145254, 0],
    #     "2D-sphere" => Float32[0.0406717, 0.0436037, 0.0111401, 0.470527, -0.310894],
    #     "3D-sphere" => Float32[-0.0483643, -0.0815794, -0.00751761, 0.54353, -0.381655],
    #     "union"     => Float32[-0.100887, -0.00164709, 0.00833292, 0.418425, -0.260235])
end

# compare across example types
crossloss = [loss(test;p,smooth! = pseudo!) for (_,p) ∈ opt, (_,test) ∈ data]
begin
    crossloss = [   -1.87665    1.16437   -0.932718   1.47818    0.514677   1.36797
                    -0.622406  -1.81191   -0.473679   0.145313  -0.233023  -0.254706
                    -1.36729    1.58016   -1.39741    1.86527    1.02292    1.77555
                    -0.576947  -1.08601   -1.12405   -1.39806   -0.750951  -1.19397
                    -1.07719   -0.969893  -1.19038   -0.711357  -1.18484   -1.04052
                    -0.584837  -1.11636   -1.10983   -1.06949   -0.865098  -1.30172
                    -1.60192   -1.48928   -1.32291   -1.33769   -1.09491   -1.21621]
    # crossloss = [ -1.55321    0.640369  -1.46382   -0.0151795   0.648754   0.390737
    #             -0.499447  -1.58266   -0.481403  -0.050292   -0.115627  -0.165515
    #             -1.37756    0.671862  -1.48931    0.0606938   0.706572   0.473284
    #             -1.40851   -1.19475   -1.44604   -1.21932    -0.766175  -0.962715
    #             -1.32879   -1.22121   -1.47734   -1.17385    -1.17231   -1.20132
    #             -0.604588  -1.24316   -1.2717    -1.14473    -0.881098  -1.23876
    #             -1.42622   -1.22249   -1.47834   -1.20185    -1.15029   -1.17799]
end

using Plots.PlotMeasures
begin
    lim = maximum(abs.(crossloss))
    names(d) = (1:length(d),keys(d))
    plot(size=(400,350))
    display(heatmap!(crossloss',c=palette([:green,:white,:red],13),clims=(-lim,lim),
            title="log₁₀ residual reduction",titlefontsize=12,
            xaxis=("train",names(opt),60),yaxis=("test",names(data),:flip)))
end
savefig("crossloss.png")

# function lerp(x,y)
#     i = floor(Int,x); dx = x-i
#     i+2>length(y) && return y[end]
#     y[i+2]*dx+y[i+1]*(1-dx)
# end

using GeometricMultigrid: mg, @loop
data = create_synthetic(len=1)
for name in ["2D-static", "2D-dipole", "2D-sphere"]
    A,b = data[name][1]
    x,hist = mg(A,b)
    name == "2D-sphere" && (@loop x[I] *= -4*A.D[I])
    heatmap(x.data[x.R],size = (250,250),legend=false,axis=nothing)
    savefig(name*".png")
end