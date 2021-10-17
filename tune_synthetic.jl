using TunedSmoother, Plots, DataStructures

# create synthetic example systems
data = create_synthetic()
udata = vcat((d[1:17] for (_,d) in data)...)
# create Parameterized preconditioner and smoother 
begin
    pmodels(p) = (D->1+p[1]+D*(p[2]+D*p[3]),L->L*(p[4]*(L-2)+p[5]*(L-1)))
    p₀ = zeros(Float32,5)
    opt = OrderedDict(name=>p₀ for name ∈ keys(data))
    opt["union"] = p₀
end

# optimize Parameterized smoother
begin
    for (name,train) ∈ data
        @show name
        opt[name] = fit(train,opt[name];pmodels,precond!)
    end
    @show "union"
    opt["union"] = fit(udata,opt["union"];pmodels,precond!)
    opt
end

begin
    using Plots
    a,b = pmodels(opt["union"])
    c = jmodel(opt["union"])
    display(plot(0:0.01:1,b,label="smoother",size=(400,400),
         xaxis=("aᵢⱼ"),yaxis=("fᵢⱼ"),legend=:bottomleft))
    plot(-6:0.1:0,c,label="preconditioner",size=(400,400),
         xaxis=("aᵢᵢ"),yaxis=("fᵢᵢ"),legend=:bottomleft)
    plot!(-6:0.1:0,a,label="smoother")
end

# compare across example types
crossloss = [loss(test;p,smooth! = pseudo!,pmodels,precond!) for (_,p) ∈ opt, (_,test) ∈ data]

begin # precond! = Jacobi!
    opt = OrderedDict(
        "2D-static" => Float32[-2.46273, -1.32142, -0.175033, -0.145183, 0.0],
        "3D-static" => Float32[0.935289, 0.210679, 0.00503365, -0.32304, 0.0],
        "2D-dipole" => Float32[2.44271, 1.50327, 0.227535, -0.165192, 0.0],
        "3D-dipole" => Float32[-0.471241, -0.155289, -0.00746283, -0.145689, 0.0],
        "2D-sphere" => Float32[0.0418186, 0.0406143, 0.0104014, -0.160154, -0.167424],
        "3D-sphere" => Float32[-0.0469152, -0.0825253, -0.00776483, -0.159348, -0.233695],
        "union"     => Float32[-0.0979022, 0.000875709, 0.00877584, -0.157031, -0.116288]
    )
    crossloss = [ -1.55321    0.926632   -1.55274   0.208985    0.51018    0.556314
                -0.289143  -1.59594    -0.46492  -0.0133369   1.96071    1.35512
                -1.33395    2.07552    -1.58787   0.829439    0.115399   1.38073
                -1.44998   -1.20046    -1.55775  -1.30888    -1.09139   -1.12327
                -1.31764    0.0917442  -1.57651  -0.393616   -1.17564   -0.167176
                -0.519564  -1.24897    -1.34205  -1.23944    -0.826082  -1.23331
                -1.43735   -1.24425    -1.5742   -1.29866    -1.14238   -1.16273  
    ]
end

using Plots.PlotMeasures
begin
    lim = 1.83 #-minimum(crossloss)
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