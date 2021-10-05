using TunedSmoother,BenchmarkTools,Plots
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
        scaledata = scale==4 ? data : create_waterlily(p=scale+2)
        opt = OrderedDict(name=>fit(train,opt[name]) for (name,train) ∈ scaledata)
        scaleloss[scale+1,:] .= [loss(test;p=opt[name],smooth!) for (name,test) ∈ data]
    end
end

begin # output from the block above
    opt = OrderedDict(
        circle => [0.0271442, -0.0248311, -0.000622211, 0.651889, -0.476478, 1.9026, 1.28664, 0.24255, 0.0105331],
        TGV    => [-0.188616, -0.0378252, 0.00345329, 0.39165, -0.260235, -3.87929, -2.60139, -0.538449, -0.0352438],
        donut  => [-0.116905, -0.0244976, 0.00293478, 0.274545, -0.148386, -0.579809, -0.771899, -0.19491, -0.0143433],
        wing   => [-0.21447, -0.245247, -0.041811, 0.790682, -0.613002, -0.101302, -0.447094, -0.350419, -0.0669029],
        shark  => [-0.263364, -0.270855, -0.0459526, 0.824633, -0.653598, -0.0514562, -0.166805, -0.145901, -0.0262677])

    scaleloss= [-1.75988  -1.52343  -1.48713  -1.37156  -1.45274
                -1.74066  -1.69835  -1.63697  -1.4343   -1.46331
                -1.78684  -1.73532  -1.6444   -1.4343   -1.47126
                -1.75631  -1.73744  -1.65368  -1.43682  -1.47672
                -1.80746  -1.73826  -1.65799  -1.43997  -1.50454]
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

# plot best pseudo-inverse functions across examples
begin
    using Plots
    models=p->(D->1+p[1]+D*(p[2]+D*p[3]),L->L*(p[4]+L*p[5]))
    a,_ = models(p₀)
    plot(-6:0.1:0,a,label="transfer",size=(400,400),
         xaxis=("aᵢᵢ"),yaxis=("fᵢᵢ"),legend=:bottomleft)
    for (name,p) in opt
        x = name ∈ (TGV,donut) ? (-6:0.1:0) : (-4:0.1:0)
        a,_ = models(p)
        plot!(x,a,label=name)
    end
    display(plot!(-6:0,i->1,label="jacobi",c=:grey))
    savefig("diag_fun.png")

    x = 0:0.02:1
    _,a = models(p₀)
    plot(x,a,label="transfer",size=(400,400),
         xaxis=("aᵢⱼ"),yaxis=("fᵢⱼ"),legend=:bottomright)
    for (name,p) in opt
        _,a = models(p)
        plot!(x,a,label=name)
    end
    display(plot!(x,i->0,label="jacobi",c=:grey))
    savefig("lower_fun.png")
end

begin
    scaledata = make_data(p=4)
    scaledata[wing] = filter(d->itcount(d;smooth! =GS!)<32,scaledata[wing])
    opt2 = OrderedDict(name=>fit(train;it=2) for (name,train) ∈ scaledata)
end
begin
    opt2 = OrderedDict(
        circle => [0.412175, 0.288861, 0.0510388, 0.72032, -0.554927],
        TGV    => [-0.447214, -0.169634, -0.00902928, 0.349716, -0.2018],
        donut  => [0.162644, 0.0423348, 0.00964613, 0.347914, -0.198642],
        wing   => [-0.433754, -0.436679, -0.0765471, 0.703688, -0.523553],
        shark  => [0.124323, -0.063333, -0.0169111, 0.660478, -0.474286])
end

# time a single Vcycle
begin
    using 
    temp!(st;kw...) = mg!(st;inner=2,mxiter=1,kw...)
    jacobi_time = @belapsed temp!(st;smooth! = Jacobi!) setup=(st=state($data[shark][1]...)) #
    gauss_time = @belapsed temp!(st;smooth! = GS!) setup=(st=state($data[shark][1]...)) #
    sor_time = @belapsed temp!(st;smooth! = SOR!) setup=(st=state($data[shark][1]...)) #
    pseudo_time = @belapsed temp!(st) setup=(st=state($data[shark][1]...)) #
    gauss_time /= jacobi_time
    sor_time /= jacobi_time
    pseudo_time /= jacobi_time
end

# Count the number of cycles needed
crosscount = [[avecount(d;smooth! = GS!) for (_,d) ∈ data]'.*gauss_time
              [avecount(d;smooth! = SOR!) for (_,d) ∈ data]'.*sor_time
              [avecount(d) for (_,d) ∈ data]'.*pseudo_time
              [avecount(d;p=opt2[name]) for (name,d) ∈ data]'.*pseudo_time]
crosscount=[ 6.35661  6.3778   6.35661  11.5915   13.4125
             6.62809  6.71706  6.71706  12.6037   14.7464
             2.71573  3.44818  3.63975   8.7055    8.643
             2.29879  2.37767  3.3693    7.9101    7.94435]

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

begin
    for (name,h) in ((circle,160),(wing,300),(shark,160))
        st = state(data[name][end]...)
        lim = maximum(st.r)
        st.x.data .-= st.x[1]
        plot(heatmap(st.A.D[st.x.R]',legend=false,c=:Blues),
            heatmap(st.x.data[st.x.R]',legend=false,clims=(-1,1)),
            heatmap(st.r.data[st.x.R]',legend=false,c=:RdBu_11,clims=(-0.1*lim,0.1*lim)),
            layout = (1,3),size=(900,h),axis=nothing)
        savefig(string(name)*"triple.png")
    end
    for (name,h,i) in ((TGV,300,25),)
        st = state(data[name][i]...)
        lim = maximum(st.r)
        st.x.data .-= st.x[1]
        plot(heatmap(st.A.D[st.x.R[:,:,end÷4]]',legend=false,c=:Blues),
            heatmap(st.x.data[st.x.R[:,:,end÷4]]',legend=false),
            heatmap(st.r.data[st.x.R[:,:,end÷4]]',legend=false,c=:RdBu_11,clims=(-0.1*lim,0.1*lim)),
            layout = (1,3),aspect_ratio=:equal,size=(900,h),axis=nothing)
        savefig(string(name)*"triple.png")
    end
    for (name,h,i) in ((donut,160,100),)
        st = state(data[name][i]...)
        lim = maximum(st.r)
        st.x.data .-= st.x[1]
        plot(heatmap(st.A.D[st.x.R[:,:,end÷4]]',legend=false,c=:Blues),
            heatmap(st.x.data[st.x.R[:,:,end÷4]]',legend=false,clims=(-1,1)),
            heatmap(st.r.data[st.x.R[:,:,end÷4]]',legend=false,c=:RdBu_11,clims=(-0.1*lim,0.1*lim)),
            layout = (1,3),aspect_ratio=:equal,size=(900,h),axis=nothing)
        savefig(string(name)*"triple.png")
    end
end
