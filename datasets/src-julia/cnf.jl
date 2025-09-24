"""
    CNF(clauses::Vector{Vector{Int}})

A type representing a conjunctive normal form.
The constructor takes a vector of clauses, where each clause is a vector of integers.
The integers represent the variables, and their sign represents the negation.
For example, the clause `x1 ∨ ¬x2 ∨ x3` is represented as `[1, -2, 3]`.
"""
struct CNF
    N::Int
    M::Int
    clauses::Vector{Vector{Int}}
end

function CNF(clauses::Vector{Vector{Int}})
    M = length(clauses)
    N = maximum(maximum(abs.(c)) for c in clauses)
    return CNF(N, M, clauses)
end

"""
    add_clause!(cnf::CNF, c::Vector{Int})

Adds a clause `c` to `cnf`.
"""
function add_clause!(cnf::CNF, c::Vector{Int})
    push!(cnf.clauses, c)
    cnf.M += 1
end

"""
    readcnf(fname::String)

Reads a CNF from file `fname`.
"""
function readcnf(fname::String)
    f = open(fname, "r")
    head = split(readline(f))
    N, M = parse(Int, head[3]), parse(Int, head[4])
    clauses = Vector{Vector{Int}}()
    for i=1:M
        line = readline(f)
        c = [parse(Int64, e) for e in split(line)[1:end-1]]
        push!(clauses, c)
    end
    return CNF(N, M, clauses)
end


"""
    writecnf(fname::String, cnf::CNF)

Writes `cnf` to file `fname`.
"""
function writecnf(fname::String, cnf::CNF)
    f = open(fname, "w")
    println(f, "p cnf $(cnf.N) $(cnf.M)")
    for c in cnf.clauses
        for i in c
            print(f, i, " ")
        end
        print(f, "0\n")
    end
    close(f)
end

"""
    energy(cnf, σ)

Counts the number of violated clauses.
"""
function energy(cnf::CNF, σ::Vector{Int})
    E = 0
    for c in cnf.clauses
        issatisfied = false
        for i in c
            if sign(i) == σ[abs(i)]
                issatisfied = true
                break
            end
        end
        E += issatisfied ? 0 : 1
    end
    E
end

"""
    adjlist(cnf::CNF)

Returns a vector containing for each variable a vector of the adjacent clause indexes.
"""
function adjlist(cnf::CNF)
    adj = Vector{Vector{Int}}(cnf.N)
    for (μ, c) in enumerate(cnf.clauses)
        for i in c
            push!(adj[abs(i)], μ)
        end
    end
end

function to_edge_index(cnf::CNF)
    N = cnf.N
    srcV, dstF = Vector{Int}(), Vector{Int}()
    srcF, dstV = Vector{Int}(), Vector{Int}()
    for (a, c) in enumerate(cnf.clauses)
        for v in c
            negated = v < 0
            push!(srcV, abs(v) + N*negated)
            push!(dstF, a)
            push!(srcF, a)
            push!(dstV, abs(v) + N*negated)
        end
    end
    return srcV, dstF, srcV, dstF
end

function to_adjacency_matrix(cnf::CNF)
    M, N = cnf.M, cnf.N
    A = spzeros(Int, M, 2*N)
    for (a, c) in enumerate(cnf.clauses)
        for v in c
            negated = v < 0
            A[a, abs(v) + N*negated] = 1
        end
    end
    return A
end