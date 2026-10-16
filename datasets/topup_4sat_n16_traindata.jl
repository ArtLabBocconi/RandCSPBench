using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using StableRNGs
include("src-julia/SAT.jl")
using .SAT: randomcnf, writecnf

# Top-up for the published kSAT.zip's 4SAT/train set.
#
# For N=16, k=4, the train alpha grid 9:0.05:10 (21 values) is finer than
# round(N*alpha) can resolve: M=144..160, but 4 of the 21 alphas collide
# with a neighbour on the same integer M (M in {146,150,154,158}, each hit
# by 2 alphas). The generator previously in use reset the id counter to
# 1:800 on every alpha, so the second colliding alpha silently overwrote
# the first alpha's 800 files on disk before they were ever published.
# Net effect: those 4 M's only ever got 800 published samples instead of
# the intended 1600, i.e. 3200 "missing" instances for N=16 train
# (13600 published instead of 16800).
#
# gen_graphs_sat.jl has since been fixed to accumulate ids across
# colliding alphas instead of overwriting (see generate_random_cnfs).
# But replaying the full alpha loop with that fix does NOT reproduce the
# already-published ids 1:800 byte-for-byte: it assigns ids in RNG-draw
# order, so ids 1:800 would become the *first* colliding alpha's draws
# (never published, now unrecoverable) while the *last* colliding alpha's
# draws (= what's currently published as ids 1:800) would shift to
# ids 801:1600.
#
# So instead of replaying, this script tops up each affected M bucket
# in place: keep the published ids 1:800 exactly as they are, and
# generate 800 brand-new samples per bucket as ids 801:1600, using a
# dedicated RNG seed (10_000_000 + N) that is clearly outside the range
# any (seed, N) pair in the main script could ever produce (seed=17,
# N <= 16384 => max 16401), so it can never collide with or be confused
# for a "real" generation seed.

N = 16
k = 4
TOPUP_SEED = 10_000_000 + N
COLLIDING_MS = [146, 150, 154, 158]
TOPUP_SAMPLES = 800
OUTDIR = ARGS[1]

rng = StableRNGs.StableRNG(TOPUP_SEED)
for M in COLLIDING_MS
    α = M / N
    for id in (TOPUP_SAMPLES + 1):(2 * TOPUP_SAMPLES)
        cnf = randomcnf(; N, α, k, rng)
        @assert cnf.M == M "expected M=$M, got $(cnf.M) (α=$α)"
        writecnf(joinpath(OUTDIR, "N$(N)_M$(M)_id$(id).cnf"), cnf)
    end
    println("M=$M: wrote ids $(TOPUP_SAMPLES+1):$(2*TOPUP_SAMPLES)")
end
println("Top-up generation complete.")
