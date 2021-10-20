using TunedSmoother, Plots, DataStructures

# create synthetic example systems
data = create_synthetic()
udata = vcat((d[1:17] for (_,d) in data)...)
# create Parameterized preconditioner and smoother 
begin
    pre = Jacobi!
    pmodels(p) = (D->1+p[1]+D*(p[2]+D*p[3]),L->L*(p[4]*(L-2)+p[5]*(L-1)))
    p₀ = zeros(Float32,5)
    opt = OrderedDict(name=>p₀ for name ∈ ["Jacobi" keys(data)... "union"])
end

# optimize Parameterized smoother
begin
    for (name,train) ∈ data
        @show name
        opt[name] = fit(train,opt[name];pmodels,precond! =pre)
    end
    @show "union"
    opt["union"] = fit(udata,opt["union"];pmodels,precond! =pre)
    opt
end
# name = "3D-static"; opt[name] = fit(data[name],opt[name];pmodels,precond! =pre)

# compare across example types
crossloss = [loss(test;p,smooth! = pseudo!,pmodels,precond! = pre) for (_,p) ∈ opt, (_,test) ∈ data]

begin # pre = Jacobi!
    opt = OrderedDict( 
        "Jacobi"    => [0.0, 0.0, 0.0, 0.0, 0.0],
        "2D-static" => [-2.46273, -1.32142, -0.175033, -0.145183, 0.0],
        "3D-static" => [0.935464, 0.210724, 0.00503401, -0.323074, 0.0],
        "2D-dipole" => [-0.00244387, 0.00340375, 0.00547249, -0.164399, 0.0],
        "3D-dipole" => [-0.00239282, 0.00309743, 0.00731025, -0.163975, 0.0],
        "2D-sphere" => [0.030264, 0.0406226, 0.0110177, -0.158986, -0.145829],
        "3D-sphere" => [-0.0477943, -0.0800523, -0.00728708, -0.160163, -0.222702],
        "union"     => [-0.104447, -0.00238399, 0.00841367, -0.158046, -0.115103],
    )
    crossloss = [-0.149117  -0.0530345  -0.572381  -0.751236    -0.17155   -0.447677
                 -1.55321    0.640408  -1.64163   -0.0148446   0.557276   0.427443
                -0.499442  -1.58268   -0.519982   0.0663901  -0.159964  -0.137229
                -1.17578   -1.2567    -1.68493   -1.42541    -1.15548   -1.18972
                -0.985601  -1.26816   -1.65395   -1.53271    -1.09258   -1.16261
                -1.33663   -1.22037   -1.67039   -1.50039    -1.19439   -1.18552
                -0.609577  -1.23985   -1.45963   -1.46347    -0.888622  -1.224
                -1.42595   -1.22271   -1.66674   -1.54645    -1.17277   -1.16282
                ]
    crossloss[2:end,:]./crossloss[1,:]'
end


using Plots.PlotMeasures
begin
    lim = -minimum(crossloss)
    names(d) = (1:length(d),keys(d))
    plot(size=(450,350))
    display(heatmap!(crossloss',c=palette([:green,:white,:red],17),clims=(-lim,lim),
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