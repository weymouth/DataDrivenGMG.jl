module TunedSmoother

include("candidate_smoothers.jl")
export Jacobi!,GS!,SOR!,pseudo!,state,precond!,tunedJ!

include("create_synthetic.jl")
export create_synthetic,static,dipole,sphere

include("create_waterlily.jl")
export create_waterlily,circle,TGV,donut,wing,shark

include("metrics.jl")
export itcount,avecount,loss,fit,mg!

end # module
