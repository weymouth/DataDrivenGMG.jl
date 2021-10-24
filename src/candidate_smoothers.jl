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
@fastmath function SOR!(st;inner=3,ω=0.9,kw...)
    @inbounds for I ∈ st.iD.R  # optimized first iteration
        st.ϵ[I] = ω*st.iD[I]*(st.r[I]-multL(I,st.A.L,st.ϵ))
     end
     @inbounds for _ ∈ 2:inner, I ∈ st.iD.R
       st.ϵ[I] = (1-ω)*st.ϵ[I]+ω*st.iD[I]*(st.r[I]-multL(I,st.A.L,st.ϵ)-multU(I,st.A.L,st.ϵ))
    end
    increment!(st;kw...)
end
@fastmath Jacobi!(st;inner=1,kw...) = for _ ∈ 1:inner
    @loop st.ϵ[I] = st.r[I]*st.iD[I]
    increment!(st;kw...)
end

# Create parameterized MG state
function state(A,x,b;p=p₀,xT::Type=eltype(p),smooth! =GS!,damp=false,kw...)
    y = zero(x,xT)     # make FieldVector of type xT
    @loop y[I] = x[I]  # copy values
    err = sum(b)/length(b)
    @loop b[I] -= err
    st = mg_state(A,y,b)
    smooth! == pseudo! && fill_pseudo!(st;p,kw...)
    damp && adjust_iD!(st;p)
    st
end
state(A,b;kw...) = state(A,zero(b),b;kw...)
fill_pseudo!(st;kw...) = !isnothing(st.child) && (st.P=PseudoInv(st.A;kw...); fill_pseudo!(st.child;kw...))
adjust_iD!(st;p) = !isnothing(st.child) && (@loop st.iD[I] *= p[end]; adjust_iD!(st.child;p))

# Apply smoother
@inline pseudo!(st;kw...) = pseudo!(st,st.P;kw...)
@fastmath pseudo!(st,P::FieldMatrix;inner=2,kw...) = for _=1:inner
    @loop st.ϵ[I] = mult(I,P.L,P.D,st.r)
    increment!(st)
end

# Parameterized approximate inverse matrix P
p₀ = Float32[-0.104447, -0.00238399, 0.00841367, -0.158046, -0.115103] # result from tune_synthetic
function PseudoInv(A::FieldMatrix; scale=maximum(A.L),p::AbstractVector{T}=p₀,
    pmodels=p->(D->1+p[1]+D*(p[2]+D*p[3]),L->L*(p[4]*(L-2)+p[5]*(L-1))),kw...) where T

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

