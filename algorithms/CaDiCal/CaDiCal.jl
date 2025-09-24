module CaDiCaL

"""
    CaDiCaL.Solver([exepath::String]; limit::Int=600)

Create a Cadical solver object. The solver is created with the given executable path.

# Arguments

- `exepath::String`: Path to the Cadical executable. 
                     If not provided, the default path `build/cadical` is used.

- `limit::Int`: Time limit in seconds for the solver. Default is 600 seconds.

# Examples

```julia
solver = CaDiCal.Solver("path/to/cadicalexecutable")
cnf_path = "path/to/cnf"
issat, sol = CaDiCal.solve(solver, cnf_path)
```
"""
struct Solver
    exe::String
    limit::Int
end

Solver(; limit::Int=600) = Solver(joinpath(@__DIR__, "build/cadical"); limit)

function Solver(exe; limit::Int=600)
    return Solver(exe, limit)
end

function solve(cadical::Solver, cnf_path::String)
    cmd = Cmd(`$(cadical.exe) -t $(cadical.limit) $(cnf_path)`, ignorestatus=true)
    out = read(cmd, String)
    lines = split(out,'\n')
    idx_s = findfirst(l -> startswith(l, "s "), lines)
    s_string = lines[idx_s]
    lines = lines[idx_s+1:end]
    seconds = grep_time(lines)

    if s_string == "s SATISFIABLE"
        issat = true
        sol = Int[]  # TODO allocate at once
        for l in lines
            if startswith(l, "v ")
                append!(sol, [parse(Int, si) for si in split(l[3:end], ' ')]) # TODO read without splitting
            else
                break
            end
        end
        @assert sol[end] == 0
        pop!(sol)
        return (; issat, sol = sign.(sol), seconds)
    elseif s_string == "s UNSATISFIABLE"
        issat = false
        return (; issat, sol=Int[], seconds)
    else
        issat = missing
        return (; issat, sol=Int[], seconds)
    end
end

function grep_time(lines)
    idx = findfirst(l -> startswith(l, "c total real time"), lines)
    l = lines[idx]

    # Regex to find a float number
    regex = r"-?\d+(\.\d+)?(e-?\d+)?"

    # Match the first floating point number
    m = match(regex, l)
    return parse(Float64, m.match)
end


end # module