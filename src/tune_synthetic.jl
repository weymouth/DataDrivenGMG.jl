using GeometricMultigrid, Plots, Printf, Optim

# create synthetic example systems
begin
    include("data_synthetic.jl")
    data = make_data()
end

# measure loss against examples
begin
    fill_pseudo!(st;kw...) = fill_pseudo!(st,st.child;kw...)
    fill_pseudo!(st,child;kw...) = (fill_pseudo!(st,nothing;kw...);fill_pseudo!(child;kw...))
    fill_pseudo!(st,::Nothing;kw...) = st.P=PseudoInv(st.A;kw...)

    function predict(A,b;p::AbstractVector{T},kw...) where T
        st = mg_state(A,zero(b,T),b)
        fill_pseudo!(st;p,kw...)
        h = mg!(st,mxiter=1,inner=2,log=true)
        log10(h[2]/h[1])
    end
    loss(data;kw...) = sum(predict(d...;kw...) for d ∈ data)/length(data)
end

# optimize psuedo-inverse functions
begin
    models=p->(D->1+p[1]+D*(p[2]+D*p[3]),
               L->       L*(p[4]+L*p[5]))
    fit(data,models,p₀=zeros(Float32,5)) = 
        Optim.minimizer(optimize(p->loss(data;p,models),p₀, Newton(); autodiff = :forward))

    opt = OrderedDict(name=>fit(train,models) for (name,train) ∈ data)
    opt["all"] = fit(vcat((d for (_,d) in data)...),models)
    opt["jacobi"] = zeros(Float32,5)
    opt
end
# "2D-box" => [-2.36056, -1.40822, -0.203578, 0.271696, -5.47975f-7]
# "3D-box" => [0.206214, 0.0214632, -0.00525863, 0.315253, 0.0135281]
# "2D-μ"   => [-1.71116, -0.902871, -0.114227, 0.166726, 3.14462f-7]
# "3D-μ"   => [-0.663327, -0.199177, -0.00915381, 0.149958, -1.54752f-7]
# "circle" => [-0.003891, 0.047247, 0.0149267, 0.377898, -0.217221]
# "sphere" => [-0.0966329, -0.0967704, -0.00862894, 0.432134, -0.271047]
# "all"    => [-0.144944, -0.0162348, 0.00734042, 0.363533, -0.201831]
# "jacobi" => [0.0, 0.0, 0.0, 0.0, 0.0]

# compare across example types
crossloss = [loss(test;p,models) for (_,p) ∈ opt, (_,test) ∈ data]
begin
    lim = maximum(abs.(crossloss))
    names(d) = (1:length(d),keys(d))
    plot(aspect_ratio=:equal,size=(560,400))
    display(heatmap!(crossloss',c=:RdBu_9,clims=(-lim,lim),
            xaxis=("train",names(opt)),yaxis=("test",names(data),:flip)))
end
savefig("crossloss.png")

begin
    for (i,name) in enumerate(keys(data))
        b,f,j = crossloss[i,i],crossloss[end-1,i],crossloss[end,i]
        db,dj = 100*(b/f-1),100*(j/f-1)
        @printf "%8s: %0.2f | %0.2f → %2.0f%% | %0.2f → %2.0f%% \n" name f b db j dj
    end
end
#          fit  |  test=train |    jacobi
# 2D-box: -1.50 | -1.75 → 16% | -0.15 → -90% 
# 3D-box: -1.22 | -1.46 → 19% | -0.05 → -96% 
#   2D-μ: -1.63 | -1.66 →  1% | -0.57 → -65% 
#   3D-μ: -1.50 | -1.52 →  1% | -0.76 → -50% 
# circle: -1.16 | -1.18 →  2% | -0.17 → -85% 
# sphere: -1.21 | -1.26 →  5% | -0.50 → -58% 

# Recursively plot MG solutions
begin
    mymap(st) = mymap(st,st.child)
    mymap(st,child) = (mymap(child); mymap(st,nothing))
    mymap(st,::Nothing) = display(heatmap(st.x.data[st.x.R]',aspect_ratio=:equal))

    function plot_mgstate(st,recurse=false)
        display(heatmap(st.r.data[st.r.R]',aspect_ratio=:equal))
        display(heatmap(st.A.D[st.A.R]',aspect_ratio=:equal))
        display(mg!(st,mxiter=1,log=true))
        display(st)
        if recurse
            mymap(st)
        else
            mymap(st,nothing)
        end
        display(heatmap(st.r.data[st.r.R]',aspect_ratio=:equal))
    end
end

# take a look at worst case
begin
    worst(data;kw...) = data[argmax([predict(d...;kw...) for d ∈ data])]
    function plot_worst(test,train)
        A,b = worst(data[test];p=opt[train],models)
        st = mg_state(A,zero(b),b)
        fill_pseudo!(st;p=opt[train],models)
        plot_mgstate(st)
    end
end
plot_worst("2D-box","fit")

# histogram of diagonal values
begin
    diag_stats(st::SolveState) = (diag_stats(diag(st.A),maximum(st.A.L)); diag_stats(st.child))
    function diag_stats(::Nothing) end
    diag_stats(D,s,width=.25) = println([count(i->(bin<=i<bin+width), D./s) for bin ∈ -6:width:0])
    A,b = sphere(N=2); diag_stats(mg_state(A,zero(b),b))
end

function lerp(x,y)
    i = floor(Int,x); dx = x-i
    i+2>length(y) && return y[end]
    y[i+2]*dx+y[i+1]*(1-dx)
end
