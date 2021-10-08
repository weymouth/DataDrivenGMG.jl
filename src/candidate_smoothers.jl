using GeometricMultigrid
using GeometricMultigrid: multL, multU, mult, increment!, δ

# Classical smoothers
@fastmath function GS!(st;inner=3,kw...)
    @inbounds for I ∈ st.iD.R  # optimized first iteration  
        st.ϵ[I] = st.iD[I]*(st.r[I]-multL(I,st.A.L,st.ϵ))
    end
    @inbounds for _ ∈ 2:inner, I ∈ st.iD.R
        st.ϵ[I] = st.iD[I]*(st.r[I]-multL(I,st.A.L,st.ϵ)-multU(I,st.A.L,st.ϵ))
    end
    increment!(st;kw...)
end
@fastmath function SOR!(st;inner=3,ω=1.15,kw...)
    @inbounds for I ∈ st.iD.R  # optimized first iteration
        st.ϵ[I] = ω*st.iD[I]*(st.r[I]-multL(I,st.A.L,st.ϵ))
     end
     @inbounds for _ ∈ 2:inner, I ∈ st.iD.R
       st.ϵ[I] = (1-ω)*st.ϵ[I]+ω*st.iD[I]*(st.r[I]-multL(I,st.A.L,st.ϵ)-multU(I,st.A.L,st.ϵ))
    end
    increment!(st;kw...)
end
@fastmath function Jacobi!(st;kw...)
    @loop st.ϵ[I] = st.r[I]*st.iD[I] # hacked by TunedJacobi!
    increment!(st;kw...)
end

# Parameterized approximate inverse matrix P
# p₀ = Float32[-0.182866, -0.0351219, 0.00462202, 0.382929, -0.230043, 0.0644419, 0.234885, 0.0664019] # result from tune_synthetic
p₀ = zeros(Float32,12) # result from tune_synthetic
function PseudoInv(A::FieldMatrix; scale=maximum(A.L),p::AbstractVector{T}=p₀,
    pmodels=p->(D->1+p[1]+D*(p[2]+D*p[3]),L->L*(p[4]+L*p[5])),kw...) where T

    L,D,N = zeros(T,size(A.L)),zeros(T,size(A.D)),length(size(A.D))
    Dm,Lm = pmodels(p)
    for I ∈ A.R
        invD = (abs(A.D[I])<1e-8) ? 0. : inv(A.D[I])
        D[I] = invD*Dm(A.D[I]/scale)
        for i ∈ 1:N
            J = I-δ(i,N)
            invD = (abs(A.D[I]+A.D[J])<2e-8) ? 0. : 2inv(A.D[I]+A.D[J])
            L[I,i] = invD*Lm(A.L[I,i]/scale)
        end
    end
    FieldMatrix(L,D,A.R)
end
function TunedJacobi!(iD::FieldVector{T},A::FieldMatrix; scale=maximum(A.L),p::AbstractVector{T}=p₀,
    jmodel=p->(D->1+p[6]+D*(p[7]+D*p[8])),kw...) where T
    Dm = jmodel(p)
    @loop iD[I] = (abs(A.D[I])<1e-8) ? 0. : inv(A.D[I])*Dm(A.D[I]/scale)
end

# Create MG state with st.P
function state(A,x,b;p=p₀,xT::Type=eltype(p),smooth! =GS!,kw...)
    y = zero(x,xT)     # make FieldVector of type xT
    @loop y[I] = x[I]  # copy values
    err = sum(b)/length(b)
    @loop b[I] -= err
    st = mg_state(A,y,b)
    smooth! == pseudo! && fill_pseudo!(st;p,kw...)
    st
end
state(A,b;kw...) = state(A,zero(b),b;kw...)
fill_pseudo!(st;kw...) = fill_pseudo!(st,st.child;kw...)
fill_pseudo!(st,child;kw...) = (fill_pseudo!(st,nothing;kw...);fill_pseudo!(child;kw...))
fill_pseudo!(st,::Nothing;kw...) = (st.P=PseudoInv(st.A;kw...); TunedJacobi!(st.iD,st.A;kw...))

# Apply smoother
pseudo!(st;kw...) = pseudo!(st,st.P;kw...)
pseudo!(st,nothing;kw...) = nothing # Can't use pseudo! without st.P
@fastmath pseudo!(st,P::FieldMatrix;inner=2,kw...) = for _=1:inner
    @loop st.ϵ[I] = mult(I,P.L,P.D,st.r)
    increment!(st)
end

# # uniform 5-point stencil: loss ≈ -0.7, @btime(n=32)≈3.1μs
# smooth!(st,θ::AbstractVector;kw...) = smooth!(st,θ...;kw...)
# @fastmath function smooth!(st,a=1.004,b=0.159;kw...)
#     @loop st.ϵ[I] = st.iD[I]*(a*st.r[I]+b*rsum(I,st.r))
#     increment!(st;kw...)
# end
# rsum(I::CartesianIndex{N},r) where N = sum(@inbounds(r[I+δ(i,N)]+r[I-δ(i,N)]) for i in 1:N)
# θ = Float32[1.004244,0.15894948]
# loss(θ,train),loss(θ,test)
# @btime smooth!(st) setup=(st=mg_state($A,zero($b),$b))

# # uniform tensor kernel: loss ≈ -0.75. @btime(n=32)≈17.5μs
# using StaticArrays
# @inline smooth!(st,θ::AbstractVector;kw...) = smooth!(st,kern(θ...);kw...)
# @inline kern(a,b) = (v = SA[a 1 a]; b.*v'.*v)
# function smooth!(st,K::StaticArray;kw...)
#     @loop st.ϵ[I] = st.iD[I]*conv(K,@view(st.r.data[(I-oneunit(I)):(I+oneunit(I))]))
#     GeometricMultigrid.increment!(st;kw...)
# end
# function conv(K,v)
#     s = zero(eltype(K))
#     @inbounds @simd for i in eachindex(K,v)
#         s+=K[i]*v[i]
#     end
#     s
# end
# θ = Float32[0.201,1.167]
# loss(θ,train),loss(θ,test)
# @btime smooth!(st,K) setup=(st=mg_state($A,zero($b),$b);K = kern($θ...))

# # kernel with affine weights: loss≈-1.23. @btime(n=32)≈25.3μs (could speed up with precompute)
# using StaticArrays
# @inline smooth!(st,θ::AbstractVector;kw...) = smooth!(st,kern(θ...);kw...)
# @inline kern(a,b,c,d,e) = SA[a b c; d d e; b a c]
# @fastmath function smooth!(st,K::StaticArray;kw...)
#     @inline Kv(i,I) = K*SA[st.A.L[I,i],st.A.L[I+δ(i),i],1]
#     @inline Km(I) = Kv(1,I)*Kv(2,I)'
#     @loop st.ϵ[I] = -conv(Km(I),@view(st.r.data[(I-oneunit(I)):(I+oneunit(I))]))
#     GeometricMultigrid.increment!(st)
# end
# θ = Float32[-0.00985,-0.01951,0.12154,-0.07238,0.67732]
# loss(θ,train),loss(θ,test)
# @btime smooth!(st,K) setup=(st=mg_state($A,zero($b),$b);K = kern($θ...))

# using Plots
# for case in test[[rand(1:100),rand(1:100),rand(1:100)]]
#     # l,st = predict(zeros(Float32,length(θ)),case...;inner=32)
#     l,st = predict(θ,case...)
#     lim = max(maximum(st.x),-minimum(st.x))
#     mymap(st) = mymap(st,st.child)
#     mymap(st,::Nothing) = display(heatmap(st.x.data[st.x.R]',aspect_ratio=:equal,clims=(-lim,lim)))
#     mymap(st,child) = (mymap(child); mymap(st,nothing))
#     mymap(st)
#     display(heatmap(case[2][st.x.R]',aspect_ratio=:equal,clims=(-1,1)))
#     display(l)
#     display(st)
# end

# name = "affine_5p"
# begin
#     open(name*"_theta.dat", "w") do io
#         write(io,θ)
#         close(io)
#     end
#     open(name*"_hist.dat", "w") do io
#         write(io,hist)
#         close(io)
#     end
#     p = copy(θ); p.=0
#     open(io->(read!(io,p);close(io)),name*"_theta.dat")
#     p==θ
# end
