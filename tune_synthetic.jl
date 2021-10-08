using TunedSmoother, Plots, DataStructures

# create synthetic example systems
    data = create_synthetic(len=25)

# create Parameterized preconditioner and smoother 
begin
    # function lerp(x,y)
    #     i = floor(Int,x); dx = x-i
    #     i+2>length(y) && return y[end]
    #     y[i+2]*dx+y[i+1]*(1-dx)
    # end
    hermite(t,p)=(2t^3-3t^2+1)*p[1]+(t^3-2t^2+t)*p[2]+(-2t^3+3t^2)p[3]+(t^3-t^2)*p[4]
    pmodels(p) = (D->1+hermite(-D/6,p[1:4]),L->hermite(L,p[5:8]))
    jmodel(p) = D->1+hermite(-D/6,p[9:12])
    p₀ = zeros(Float32,12)
    opt = OrderedDict(name=>p₀ for name ∈ keys(data))
    opt["union"] = p₀
end

# optimize Parameterized smoother
begin
    for (name,train) ∈ data
        @show name
        opt[name] = fit(train,opt[name];pmodels,jmodel)
    end
    @show "union"
    opt["union"] = fit(vcat((d for (_,d) in data)...),opt["union"];pmodels,jmodel)
    opt
end
begin
    opt = OrderedDict(
        "2D-static" => [-2.26078, -1.35812, -0.198977, 0.467383, -0.260235, 0.0687914, 0.0520279, -0.000723798, 0.00639344],
        "3D-static" => [0.806427, 0.1904, 0.0066247, 0.554321, -0.260235, -0.650988, -0.241448, -0.0228976, -0.00349145],
        "2D-dipole" => [2.80018, 1.71771, 0.25784, 0.419634, -0.260235, 0.803588, 0.530899, 0.00125979, -0.0214498],
        "3D-dipole" => [0.115248, -0.000663957, 0.000381931, 0.389719, -0.260235, 0.0322515, 0.101498, 0.00717328, -0.00287373],
        "2D-sphere" => [0.0915379, 0.0833613, 0.017635, 0.447855, -0.291, 0.687612, 1.31916, 0.695357, 0.0940616],
        "3D-sphere" => [-0.00827322, -0.0546565, -0.00438196, 0.53397, -0.377591, 0.446063, -0.0336524, -0.0570324, -0.0110678],
        "union"     => [-0.182866, -0.0351219, 0.00462202, 0.382929, -0.230043, 0.0644419, 0.234885, 0.0664019])
end

# compare across example types
crossloss = [loss(test;p,smooth! = pseudo!,pmodels,jmodel) for (_,p) ∈ opt, (_,test) ∈ data]
begin
    crossloss = [   -1.87665    1.16439   -1.13973   1.65732   -0.0803689   1.39344
                    -0.622401  -1.8119    -0.56231   0.28911   -0.428983   -0.261036
                    -1.36738    1.58017   -1.66168   2.09321    0.170027    1.68013
                    -0.576945  -1.086     -1.33573  -1.60083   -0.964724   -1.15003
                    -1.07719   -0.972905  -1.4157   -0.808376  -1.28669    -1.04982
                    -0.584835  -1.11912   -1.32193  -1.14577   -1.11273    -1.27706
                    -1.60193   -1.48923   -1.58166  -1.52131   -1.21837    -1.16797
   ]
end

using Plots.PlotMeasures
begin
    lim = -minimum(crossloss)
    names(d) = (1:length(d),keys(d))
    plot(size=(400,350))
    display(heatmap!(crossloss',c=palette([:green,:white,:red],13),clims=(-lim,lim),
            title="log₁₀ residual reduction",titlefontsize=12,
            xaxis=("train",names(opt),60),yaxis=("test",names(data),:flip)))
end
savefig("crossloss.png")

using GeometricMultigrid: mg, @loop
data = create_synthetic(len=1)
for name in ["2D-static", "2D-dipole", "2D-sphere"]
    A,b = data[name][1]
    x,hist = mg(A,b)
    name == "2D-sphere" && (@loop x[I] *= -4*A.D[I])
    heatmap(x.data[x.R],size = (250,250),legend=false,axis=nothing)
    savefig(name*".png")
end