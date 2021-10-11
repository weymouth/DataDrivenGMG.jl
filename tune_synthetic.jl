using TunedSmoother, Plots, DataStructures

# create synthetic example systems
data = create_synthetic()

# create Parameterized preconditioner and smoother 
begin
    pmodels(p) = (D->1+p[1]+D*(p[2]+D*p[3]),L->L*(p[4]+L*p[5]))
    jmodel(p) = D->1+p[6]+D*(p[7]+D*p[8])
    p₀ = zeros(Float32,8)
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
    opt["union"] = fit(vcat((d[1:17] for (_,d) in data)...),opt["union"];pmodels,jmodel)
    opt
end
begin
    opt = OrderedDict(
        "2D-static" => Float32[-2.26078, -1.35812, -0.198978, 0.207148, -2.52246f-7, 0.380395, 0.273338, 0.0192435],
        "3D-static" => Float32[0.806225, 0.190341, 0.00662217, 0.294064, -9.57746f-8, -0.437093, -0.075387, 0.019219],
        "2D-dipole" => Float32[-1.69299, -0.909276, -0.117421, 0.16275, -4.51364f-8, -0.554424, -0.328334, -0.0425878],
        "3D-dipole" => Float32[9.18217f-5, -0.000531712, 0.00308433, 0.0623805, 0.0623805, 0.00022871, -0.00152921, 0.00964473],
        "2D-sphere" => Float32[0.0402841, 0.052108, 0.0132633, 0.447416, -0.290573, -0.144877, -0.137151, -0.00147627],
        "3D-sphere" => Float32[-0.0343149, -0.0675936, -0.00572602, 0.517866, -0.372514, 0.358447, 0.12475, 0.0276228],
        "union"     => Float32[-0.0162363, 0.0304057, 0.0112845, 0.0812823, 0.0784739, -0.00240444, 0.00291828, 0.0062895])
        # "2D-static" => Float32[-2.26078, -1.35812, -0.198977, 0.207147, 3.02243f-7, 1.44879, 1.16366, 0.197309],
        # "3D-static" => Float32[-8.36772, -3.27071, -0.319728, 0.258874, 3.29758f-7, 8.14383, 4.33496, 0.439617],
        # "2D-dipole" => Float32[-0.00318763, 0.00474554, 0.00503566, 0.0802189, 0.0802189, -8.09586f-5, -0.000609329, 0.00432618],
        # "3D-dipole" => Float32[-0.222611, -0.0574386, 0.00138187, 0.0590631, 0.0802191, -0.167646, -0.0421119, 0.00343846],
        # "2D-sphere" => Float32[0.0742412, 0.0662679, 0.014608, 0.464488, -0.306209, 0.218188, 0.252256, 0.101635],
        # "3D-sphere" => Float32[0.0115758, -0.0506081, -0.00430204, 0.480167, -0.326424, 0.599652, 0.388787, 0.0881687],
        # "union"     => Float32[-0.174781, -0.0341458, 0.00452468, 0.391123, -0.236144, 0.0380468, 0.226707, 0.0667013])
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
crossloss = [loss(test;p,smooth! = pseudo!,pmodels,jmodel) for (_,p) ∈ opt, (_,test) ∈ data]
begin
    crossloss = [ -1.87665    1.12891  -1.11411   0.204975   0.428232    0.813582
                    -0.620796  -1.81188  -0.55201   0.183951  -0.217591   -0.255869
                    -1.38066    0.16583  -1.57588  -0.368618  -0.0132151  -0.129015
                    -1.48927   -1.22064  -1.53134  -1.33055   -1.05284    -1.0667
                    -1.13702   -1.21441  -1.43377  -1.15739   -1.19431    -1.21418
                    -0.568496  -1.16778  -1.22865  -1.23207   -0.840495   -1.25162
                    -1.34752   -1.29839  -1.56139  -1.25333   -1.07717    -1.1298 
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