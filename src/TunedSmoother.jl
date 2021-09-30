module TunedSmoother

include("candidate_smoothers.jl")
export Jacobi!,GS!,SOR!,pseudo!,state

include("create_synthetic.jl")
export create_synthetic

include("create_waterlily.jl")
export create_waterlily

include("metrics.jl")
export avecount,loss,fit

end # module
