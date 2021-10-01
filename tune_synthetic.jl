using TunedSmoother, Plots, DataStructures
using GeometricMultigrid: mg!

# create synthetic example systems
begin
    data = make_data()
    opt = OrderedDict(name=>zeros(Float32,5) for name ∈ keys(data))
    opt["union"] = zeros(Float32,5)
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
begin
    # opt = OrderedDict(
    #     "2D-static" => [-2.36056, -1.40822, -0.203578, 0.271696, -5.47975f-7],
    #     "3D-static" => [0.206214, 0.0214632, -0.00525863, 0.315253, 0.0135281],
    #     "2D-dipole" => [-1.71116, -0.902871, -0.114227, 0.166726, 3.14462f-7],
    #     "3D-dipole" => [-0.663327, -0.199177, -0.00915381, 0.149958, -1.54752f-7],
    #     "2D-sphere" => [-0.003891, 0.047247, 0.0149267, 0.377898, -0.217221],
    #     "3D-sphere" => [-0.0966329, -0.0967704, -0.00862894, 0.432134, -0.271047],
    #     "union"  => [-0.144944, -0.0162348, 0.00734042, 0.363533, -0.201831])
    opt = OrderedDict(
        "2D-static" => Float32[-2.46273, -1.32142, -0.175033, 0.145184, 0],
        "3D-static" => Float32[0.93542, 0.210713, 0.00503395, 0.309537, 0.0135281],
        "2D-dipole" => Float32[-2.56297, -1.41248, -0.188902, 0.163554, 0],
        "3D-dipole" => Float32[-0.752343, -0.258171, -0.0168188, 0.145254, 0],
        "2D-sphere" => Float32[0.0406717, 0.0436037, 0.0111401, 0.470527, -0.310894],
        "3D-sphere" => Float32[-0.0483643, -0.0815794, -0.00751761, 0.54353, -0.381655],
        "union"     => Float32[-0.100887, -0.00164709, 0.00833292, 0.418425, -0.260235])
end

# compare across example types
crossloss = [loss(test;p,smooth! = pseudo!) for (_,p) ∈ opt, (_,test) ∈ data]
begin
    crossloss = [ -1.55321    0.640369  -1.46382   -0.0151795   0.648754   0.390737
                -0.499447  -1.58266   -0.481403  -0.050292   -0.115627  -0.165515
                -1.37756    0.671862  -1.48931    0.0606938   0.706572   0.473284
                -1.40851   -1.19475   -1.44604   -1.21932    -0.766175  -0.962715
                -1.32879   -1.22121   -1.47734   -1.17385    -1.17231   -1.20132
                -0.604588  -1.24316   -1.2717    -1.14473    -0.881098  -1.23876
                -1.42622   -1.22249   -1.47834   -1.20185    -1.15029   -1.17799]
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
