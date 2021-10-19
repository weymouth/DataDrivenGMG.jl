using TunedSmoother,BenchmarkTools,Plots,DataStructures
using TunedSmoother: p₀

# Generate WaterLily simulation data
data = create_waterlily()
# Optimize the psuedo-inverse functions
begin # this block takes around 30 minutes. the next block has the results
    smooth! = pseudo!
    opt = OrderedDict(name=>p₀ for name ∈ keys(data))
    scaleloss = zeros(5,length(opt))
    scaleloss[1,:] .= [loss(test;p=p₀,smooth!) for (name,test) ∈ data]
    for scale ∈ 1:4
        @show scale
        scaledata = scale==4 ? data : create_waterlily(p=scale+2,cases=keys(data))
        opt = OrderedDict(name=>fit(train,opt[name]) for (name,train) ∈ scaledata)
        scaleloss[scale+1,:] .= [loss(test;p=opt[name],smooth!) for (name,test) ∈ data]
    end
end

begin # output from the block above
    opt = OrderedDict(
        circle => [-0.00727588, -0.102282, -0.0178857, 0.760556, -0.586334, 1.78245, 1.14314, 0.167734],
        TGV    => [-0.0617478, 0.00940791, 0.00780108, 0.391629, -0.260235, -1.16419, -0.5565, -0.0615989],
        donut  => [-0.134641, -0.0313802, 0.00234106, 0.286056, -0.159517, -0.021158, -0.213197, -0.0317727],
        wing   => [-0.229121, -0.253223, -0.042952, 0.782714, -0.604938, -0.00708945, 0.0287928, 0.00663962],
        shark  => [-0.279771, -0.281626, -0.0476559, 0.821296, -0.649919, -0.00109436, 0.055443, 0.00988565])

    scaleloss= [-1.75374  -1.52343  -1.48713  -1.37156  -1.45274
                -1.7893   -1.69872  -1.63804  -1.42732  -1.46177
                -1.81289  -1.73545  -1.64428  -1.42765  -1.46817
                -1.80914  -1.73749  -1.65463  -1.43041  -1.48582
                -1.82367  -1.73825  -1.65887  -1.4328   -1.50249]
end

# plot loss across examples and scales
begin
    using StatsPlots,CategoricalArrays
    cats = ["transfer","⅛","¼","½","1"]
    ctg = CategoricalArray(repeat(cats,inner=length(data)))
    levels!(ctg,cats)
    groupedbar(scaleloss', size = (400,400),
                group=ctg, legend=:bottomright, palette=:Greens_5,
                yaxis=("log₁₀ residual reduction",:flip),
                xaxis=("cases",(1:length(data),keys(data))),)
end
savefig("scaleloss.png")

begin
    scaledata = create_waterlily(p=4)
    scaledata[wing] = filter(d->itcount(d,GS!)<32,scaledata[wing])
    opt2 = OrderedDict(name=>fit(train;it=2) for (name,train) ∈ scaledata)
end
begin
    opt2 = OrderedDict(
        circle => [0.143874, 0.114321, 0.024359, -0.166979, -0.211477],
        TGV    => [-0.445326, -0.175383, -0.0100868, -0.147759, 0.0],
        donut  => [0.120699, 0.0239083, 0.00775287, -0.149375, -0.0753296],
        wing   => [0.0259308, -0.123091, -0.0268147, -0.174569, -0.542901],
        shark  => [0.0673034, -0.104414, -0.0241938, -0.182964, -0.298915]
    )
end

# time a single Vcycle
begin
    temp!(st;kw...) = mg!(st;inner=2,mxiter=1,kw...)
    jacobi_time = @belapsed temp!(st;smooth! = Jacobi!) setup=(st=state($data[shark][1]...)) #
    gauss_time = @belapsed temp!(st;smooth! = GS!) setup=(st=state($data[shark][1]...)) #
    sor_time = @belapsed temp!(st;smooth! = SOR!) setup=(st=state($data[shark][1]...)) #
    pseudo_time = @belapsed temp!(st) setup=(st=state($data[shark][1]...)) #
    gauss_time /= jacobi_time
    sor_time /= jacobi_time
    pseudo_time /= jacobi_time
end
gauss_time = 3.5; sor_time = 3.2; pseudo_time = 1.37

# Count the number of cycles needed
crosscount = [[avecount(d,GS!) for (_,d) ∈ data]'.*gauss_time
              [avecount(d,SOR!) for (_,d) ∈ data]'.*sor_time
              [avecount(d,pseudo!) for (_,d) ∈ data]'.*pseudo_time
              [avecount(d,pseudo!;p=opt2[name]) for (name,d) ∈ data]'.*pseudo_time]
crosscount = [  7.0   7.56    10.5     18.585   20.825
                6.4   9.632    9.632   17.92    20.128
                2.74  4.11     4.1511   8.7954  10.1106
                2.74  2.8907   4.0963   8.2063   9.453
            ]

begin
    using StatsPlots,CategoricalArrays
    n = length(data)
    cats = ["Gauss-Sidel","SOR","Ã⁻¹ transfer","Ã⁻¹ tuned-¼"]
    colors = repeat([palette(:default)[2],palette(:default)[4],:lightgreen,palette(:default)[3]],inner=n)
    ctg = CategoricalArray(repeat(cats,inner=n))
    levels!(ctg,cats)
    groupedbar(crosscount',size=(400,400),
                group=ctg, legend=:bottomright,c=colors,
                yaxis=("relative time"),
                xaxis=("cases",(1:length(data),keys(data))),)
end
savefig("crosscount.png")

# plot best pseudo-inverse functions across examples
begin
    using Plots
    p₀ = Float32[-0.104447, -0.00238399, 0.00841367, -0.158046, -0.115103]
    pmodels(p) = (D->1+p[1]+D*(p[2]+D*p[3]),L->L*(p[4]*(L-2)+p[5]*(L-1)))
    a,_ = pmodels(p₀)
    plot(-6:0.1:0,a,label="transfer",size=(400,400),
         xaxis=("aᵢᵢ"),yaxis=("fᵢᵢ"),legend=:bottomleft)
    for (name,p) in opt2
        x = name ∈ (TGV,donut) ? (-6:0.1:0) : (-4:0.1:0)
        a,_ = pmodels(p)
        plot!(x,a,label=name)
    end
    display(plot!(-6:0,i->1,label="jacobi",c=:grey))
    savefig("diag_fun.png")

    x = 0:0.02:1
    _,a = pmodels(p₀)
    plot(x,a,label="transfer",size=(400,400),
         xaxis=("aᵢⱼ"),yaxis=("fᵢⱼ"),legend=:bottomright)
    for (name,p) in opt2
        _,a = pmodels(p)
        plot!(x,a,label=name)
    end
    display(plot!(x,i->0,label="jacobi",c=:grey))
    savefig("lower_fun.png")
end

begin
    using GeometricMultigrid: Vcycle!
    # for (name,h) in ((circle,160),(wing,300),(shark,160))
    # for (name,h) in ((TGV,300),)
    for (name,h) in ((wing,300),)
    # for (name,h) in ((donut,160),)
            smooth! = pseudo!
        # st = state(data[name][25]...;smooth!)
        st = state(data[name][end]...;smooth!)
        x = copy(st.x.data)
        x .-= st.x[1]
        x[st.A.D .> -1.5] .= NaN
        r = copy(st.r)
        f(x) = @. log10(clamp(abs(x),1e-6,Inf64))
        Vcycle!(st;smooth!)
        plot(heatmap(x[st.x.R]',legend=false,clims=(-1,1)),
        heatmap(f(r.data[st.x.R]'),legend=false,c=:Reds,clims=(-6,-1)),
        heatmap(f(st.r.data[st.x.R]'),legend=false,c=:Reds,clims=(-6,-1)),
        # plot(heatmap(x[st.x.R[:,:,end÷4]]',legend=false,clims=(-1,1)),
        # plot(heatmap(x[st.x.R[:,:,end÷4]]',legend=false),
        # heatmap(f(r.data[st.x.R[:,:,end÷4]]'),legend=false,c=:Reds,clims=(-6,-1)),
        # heatmap(f(st.r.data[st.x.R[:,:,end÷4]]'),legend=false,c=:Reds,clims=(-6,-1)),
        layout = (1,3),size=(900,h),axis=nothing)
        savefig(string(name)*"triple.png")
    end
end
