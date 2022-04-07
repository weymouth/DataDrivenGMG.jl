using TunedSmoother,WaterLily,Plots
using GeometricMultigrid: Vcycle!

# shark parameters from test_waterlily
p = [0.0673015, -0.104415, -0.024194, -0.182964, -0.298915]

A,St,remeasure = 0.1,0.3,true
cycles,period,steps = 2,2A/St,120
t₀,Δt = 3,cycles*period/steps

# Set up and run simulation
sim = shark(6;A,St)
sim_step!(sim,t₀;remeasure)

# Get Poisson data over next period
data = [(TunedSmoother.sim_collect_poisson!(sim;Δt,remeasure),sim_time(sim)) for i ∈ 1:steps]

# Integrate pressure data at time t
get_forces(data;kw...) = map(data) do ((A,p,b),t)
	st = state(A,b;kw...)
	mg!(st;kw...)
	WaterLily.∮nds(st.x.data,sim.body,t*sim.L/sim.U)./(0.5*sim.L*sim.U^2)
end

begin
	reltol=1e-3
	thrust = plot(xlabel="scaled time", ylabel="scaled thrust force")
	side = plot(xlabel="scaled time", ylabel="scaled side force")
	forces = get_forces(data;reltol)
	plot!(thrust,last.(data),first.(forces),label="GS",c=palette(:default)[2])
	plot!(side,  last.(data), last.(forces),label="GS",c=palette(:default)[2])
	forces = get_forces(data;reltol,smooth! = pseudo!, p)
	plot!(thrust,last.(data),first.(forces),label="Ã⁻¹",line=:dash,c=palette(:default)[3])
	plot!(side,  last.(data), last.(forces),label="Ã⁻¹",line=:dash,c=palette(:default)[3])
end
plot(thrust,side,layout=(2,1))
savefig("forces.pdf")