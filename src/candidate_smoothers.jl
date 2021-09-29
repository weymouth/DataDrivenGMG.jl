using GeometricMultigrid
using GeometricMultigrid: GS!,pseudo!,multL,multU,increment!
Jacobi!(st;kw...) = GS!(st,inner=0)
@fastmath function SOR!(st;inner=2,ω=0.9,kw...)
    @loop st.ϵ[I] = ω*st.r[I]*st.iD[I]
    for _ ∈ 1:inner
        @loop st.ϵ[I] = (1-ω)*st.ϵ[I]+ω*st.iD[I]*(st.r[I]-multL(I,st.A.L,st.ϵ)-multU(I,st.A.L,st.ϵ))
    end
    increment!(st;kw...)
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