module SAT
using Random, Statistics, LinearAlgebra
using StatsBase

include("cnf.jl")
export CNF, add_clause!, readcnf, writecnf, energy, adjlist

include("generators.jl")
export randomcnf

end # module
