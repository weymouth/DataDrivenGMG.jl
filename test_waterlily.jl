using TunedSmoother,BenchmarkTools,Plots
using TunedSmoother: p₀

# Generate WaterLily simulation data
data = create_waterlily()

# Optimize the psuedo-inverse functions
begin # this block takes around 30 minutes. the next block has the results
    opt = OrderedDict(name=>p₀ for name ∈ keys(data))
    scaleloss = zeros(4,length(opt))
    for scale ∈ 1:4
        @show scale
        scaledata = scale==4 ? data : make_data(p=scale+2)
        opt = OrderedDict(name=>fit(train;p₀=opt[name]) for (name,train) ∈ scaledata)
        scaleloss[scale,:] .= [loss(test;p=opt[name]) for (name,test) ∈ data]
    end
end

begin # output from the block above
    opt = OrderedDict(
        circle => [0.266145, 0.176086, 0.0344038, 0.453681, -0.27995],
        TGV    => [-0.150537, -0.0375542, 0.00201316, 0.329784, -0.201799],
        donut  => [0.20243, 0.0752068, 0.0116999, 0.323697, -0.189171],
        wing   => [-0.108999, -0.193017, -0.0354521, 0.532743, -0.340341],
        shark  => [-0.282117, -0.283407, -0.0472922, 0.577397, -0.402278])

    scaleloss= [-1.73515  -1.69368  -1.63451  -1.36256  -1.46331
                -1.7428   -1.73446  -1.65058  -1.37201  -1.47126
                -1.73086  -1.73621  -1.65165  -1.37961  -1.47637
                -1.74365  -1.73657  -1.65264  -1.38306  -1.47978]        
end

# plot loss across examples and scales
begin
    using StatsPlots,CategoricalArrays
    cats = ["transfer","⅛","¼","½","1"]
    ctg = CategoricalArray(repeat(cats,inner=length(data)))
    levels!(ctg,cats)
    groupedbar([compareloss[end,:] scaleloss'], size = (400,400),
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

# begin # Optimizing SOR found ω≈0.9
#     using Optim
#     ω_loss(data;ω) = sum(Δresid(state(d...;xT=typeof(ω));inner=2,smooth! = SOR!,ω) for d ∈ data)/length(data)
#     ω_fit(data;ω₀=1.0) = Optim.minimizer(optimize(p->ω_loss(data;ω=p[1]),[ω₀], Newton(),
#                       Optim.Options(time_limit=60,show_trace=true); autodiff = :forward))[1]

#     for scale ∈ 1:3
#         @show scale
#         scaledata = scale==4 ? data : make_data(p=scale+2)
#         scaledata[wing] = filter(d->itcount(d;smooth! =GS!)<32,scaledata[wing])
#         ω_opt = OrderedDict(name=>ω_fit(train) for (name,train) ∈ scaledata)
#         @show ω_opt
#         ω_error = OrderedDict(name=>ω_loss(test;ω=ω_opt[name]) for (name,test) ∈ data)
#         @show ω_error
#     end
# end