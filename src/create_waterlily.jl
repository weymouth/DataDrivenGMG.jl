import WaterLily
using WaterLily: AutoBody, Simulation
using GeometricMultigrid: Poisson,FieldVector
using LinearAlgebra: norm2

function poisson_setup!(a)
    a.u⁰ .= a.u; a.u .= 0
    WaterLily.conv_diff!(a.f,a.u⁰,ν=a.ν)
    WaterLily.BDIM!(a); WaterLily.BC!(a.u,a.U)
    WaterLily.@inside a.σ[I] = (WaterLily.div(I,a.u)+a.σᵥ[I])/a.Δt[end]
    a.u .= a.u⁰ # roll-back!
end

function sim_collect_poisson!(sim;Δt=0.1,remeasure)
    a = sim.flow # less typing
    WaterLily.sim_step!(sim,WaterLily.sim_time(sim)+Δt;remeasure)  # ∫ over dt & get new a.p
    WaterLily.measure!(a,sim.body;t=sim.L/sim.U*WaterLily.sim_time(sim)) # get new a.μ₀
    poisson_setup!(a)                            # get new a.σ
    return Poisson(copy(a.μ₀)),FieldVector(copy(a.p)),FieldVector(copy(a.σ))
end

function circle(p=6;Re=250,n=16,m=8,T=Float64)
    radius = 2^(p-2)
    center, ν = radius*m/2, radius/Re
    body = AutoBody((x,t)->norm2(x .- center) - radius)
    return Simulation((n*radius+2,m*radius+2), [1.,0.], radius; ν, body, T)
end

function TGV(p=6;Re=1e5,T=Float64)
    L = 2^p; U = 1; ν = U*L/Re
    function uλ(i,vx)
        x,y,z = @. (vx-1.5)*π/L                # scaled coordinates
        i==1 && return -U*sin(x)*cos(y)*cos(z) # u_x
        i==2 && return  U*cos(x)*sin(y)*cos(z) # u_y
        return 0.                              # u_z
    end
    return Simulation((L+2,L+2,L+2),zeros(3),L;U,uλ,ν,T)
end

function donut(p=6;Re=1e3,T=Float64)
    n = 2^p
    center,R,r = [n/2,n/2,n/2], n/4, n/16
    ν = R/Re
    body = AutoBody() do xyz,t
        x,y,z = xyz - center
        norm2([x,norm2([y,z])-R])-r
    end
    Simulation((2n+2,n+2,n+2),[1.,0.,0.],R;ν,body,T)
end

using StaticArrays: SVector, @SMatrix
function wing(p=6;Re=250,U=1,amp=π/4,ϵ=0.5,thk=2ϵ+√2,T=Float64)
    L = 2^(p-1)
    sdf(x,t) = norm2(x .- SVector(0.,clamp(x[2],-L/2,L/2)))-thk/2
    function map(x,t)
        α = amp*cos(t*U/L); R = @SMatrix [cos(α) sin(α); -sin(α) cos(α)]
        R * (x.-SVector(3L+L*sin(t*U/L)+0.01,4L))
    end
    body = AutoBody(sdf,map)
    Simulation((6L+2,6L+2),zeros(2),L;U,ν=U*L/Re,body,ϵ,T)
end

using Interpolations
function shark(p=6;k=5.3,A=0.1,St=0.3,Re=1e4,U=1,
              width = [0.02,0.07,0.06,0.048,0.03,0.019,0.01],
              envelope = [0.2,0.21,0.23,0.4,0.88,1.0],T=Float64)
    L = 2^p
    fit = y-> scale(interpolate(y, BSpline(Quadratic(Line(OnGrid())))), range(0,1,length=length(y)))
	thk,amp = fit(width),fit(envelope)
    s(x) = clamp(x[1]/L,0,1)
    sdf(x,t) = norm2(x.-L*SVector(s(x),0.))-L*thk(s(x))
    ω = 2π*St*U/(2A*L)
    function map(x,t)
        xc = x.-L # shift origin
        return xc-SVector(0.,A*L*amp(s(xc))*sin(k*s(xc)-ω*t))
    end
    return Simulation((4L+2,2L+2),[U,0.],L;ν=U*L/Re,body=AutoBody(sdf,map),T)
end	

using DataStructures
function create_waterlily(;len=100,Δt=0.1,cases=[circle,TGV,donut,wing,shark],p=6,kw...)
    data = OrderedDict()
    for case ∈ cases
        @show case
        remeasure = case ∈ (wing,shark)
        sim = case(p;kw...)
        case==circle && WaterLily.sim_step!(sim,20)
        case == circle && WaterLily.sim_step!(sim,15)
        data[case] = [sim_collect_poisson!(sim;Δt,remeasure) for i ∈ 1:len]
    end
    return data
end
