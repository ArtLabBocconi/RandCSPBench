
"""
    randomcnf(; N=100, k=3, α=0.1, rng, planted=nothing, q_planted=1)

Generates a random instance of the k-SAT problem, with `N` variables
and `α * N` clauses.

`rng` is randon numer generator.

If a vector `planted` of ±1 elements is passed, 
it is guaranteed to be a solution of the generated problem.

In the planted setting, you can also pass a coefficient `q_planted`,
beetween 0 and 1, such that a new clause satisfying the planted configuration 
is added only with probability `q^num_sat_literals` according to the prescription in Ref. [1].

[1] Haixia Jia, Cristopher Moore, and Doug Strain. “Generating hard satisfiable formulas by hiding solutions deceptiveily”. In: Proceedings of the 20th national conference on Artificial intelligence 2005.
"""
function randomcnf(; N::Int = 100, k::Int = 3, α::Float64 = 0.1, 
                    rng = Random.GLOBAL_RNG,
                    planted = nothing, 
                    q_planted = 1)

    M = round(Int, N*α)
    @assert k < N
    if planted !== nothing
        @assert length(planted) == N  "Wrong size for planted configurations ($N != $(length(planted)) )"
        @assert sort(union(planted)) == [-1, 1]
    end
    if planted !== nothing
        if q_planted < 1
            clauses = [_generate_clause(rng, N, k, planted, q_planted) for a=1:M]
        else
            clauses = [_generate_clause(rng, N, k, planted) for a=1:M]
        end
    else
        clauses = [_generate_clause(rng, N, k) for a=1:M]
    end
    return CNF(N, M, clauses)
end

function _generate_clause(rng, N, k, planted::AbstractVector, q)
    c = sample(rng, 1:N, k, replace=false)
    signs = Vector{Bool}(undef, k)
    while true
        rand!(rng, signs)
        num_sat = sum(signs)
        num_sat == 0 && continue
        # Accept with prob to q^(num_sat-1) 
        # Turns out to be equivalent to accept with prob. q^(num_sat)
        # since an overall factor those shift probabilities.
        # So  we gain a factor of 1/q speedup.
        rand(rng) < q^(num_sat-1) && break 
    end
    return c .* planted[c] .* (2 .* signs .- 1)
end

function _generate_clause(rng, N, k, planted::AbstractVector)
    c = sample(rng, 1:N, k, replace=false)
    signs = Vector{Bool}(undef, k)
    while true
        rand!(rng, signs)
        signs = rand(rng, Bool, k)
        any(signs) && break
    end
    return c .* planted[c] .* (2 .* signs .- 1)
end

function _generate_clause(rng, N, k)
    while true
        c = rand(rng, 1:N, k)
        length(union(c)) != k && continue
        return c .* rand(rng, [-1,1], k)
    end
end

# # TODO replace with this in the next release of the dataset
# function _generate_clause(rng, N, k)
#     c = sample(rng, 1:N, k, replace=false)
#     return  c .* rand(rng, [-1,1], k)
# end

