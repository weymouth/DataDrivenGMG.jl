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
begin # precond! = diagonal only
    opt = OrderedDict(
        "2D-static" => Float32[-1.03953, -0.393951, 0.00262573, 0.00874249, -0.14665, 0.0],
        "3D-static" => Float32[-3.00862, -2.22623, -0.489721, -0.0330752, -0.32141, 0.0],
        "2D-dipole" => Float32[-1.25025, -0.459942, 0.0290695, 0.0153954, -0.163855, 0.0],
        "3D-dipole" => Float32[-0.286709, -0.0252308, 0.0206485, 0.00199961, -0.140484, 0.0],
        "2D-sphere" => Float32[-0.0983752, -0.226187, -0.108934, -0.0154183, -0.160565, -0.196976],
        "3D-sphere" => Float32[-0.0908512, -0.173352, -0.039595, -0.00301273, -0.158848, -0.219527],
        "union"     => Float32[0.282169, 0.371521, 0.11341, 0.00893918, -0.151586, -0.092342]
    )
    crossloss = [ -1.70152   -0.134349  -1.56018   -0.728549  -0.304495  -0.404324
                -0.507682  -1.57548   -0.528379  -0.201006  -0.259648  -0.231753
                -1.27355    1.01485   -1.60315    0.896996   0.605993   0.957853
                -1.6071    -1.21253   -1.54016   -1.32706   -0.869969  -1.01347
                -1.28313   -1.28115   -1.58882   -1.20861   -1.17523   -1.19899
                -0.557009  -1.29971   -1.32476   -1.2153    -0.862389  -1.2427
                -1.49448   -1.27351   -1.58144   -1.29748   -1.11844   -1.15658
   ]
end
begin # precond! = pseudo!(inner=1)
    opt = OrderedDict(
        "2D-static" => Float32[-2.12829, -1.7026, -0.484515, -0.0482328, -0.144186, 0.0],
        "3D-static" => Float32[-3.61556, -2.39375, -0.504541, -0.0346249, -0.230586, 0.0],
        "2D-dipole" => Float32[-1.56016, -0.796701, -0.0842826, 0.00336884, -0.166041, 0.0],
        "3D-dipole" => Float32[-0.173378, -0.0103099, 0.0155066, 0.00122721, -0.150223, 0.0],
        "2D-sphere" => Float32[-0.0675424, -0.217462, -0.112087, -0.0162422, -0.159451, -0.219328],
        "3D-sphere" => Float32[-0.063851, -0.1578, -0.0372874, -0.00293336, -0.158647, -0.199525],
        "union"     => Float32[0.289731, 0.36828, 0.110953, 0.00865763, -0.157812, -0.0780845]
    )
    crossloss = [ -1.83857   -0.500421  -1.7541   -0.913814  -0.652567  -0.764484
                    -0.933966  -1.80007   -1.17655  -0.462506  -0.743554  -0.789049
                    -1.38506    0.405513  -1.83768   0.1119    -0.420833   0.105931
                    -1.65284   -1.51394   -1.79709  -1.54418   -1.09039   -1.19285
                    -1.32782   -1.53546   -1.82018  -1.5164    -1.23274   -1.27934
                    -0.578228  -1.42292   -1.471    -1.49241   -0.90487   -1.33764
                    -1.61188   -1.54903   -1.81306  -1.52894   -1.15492   -1.24307
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