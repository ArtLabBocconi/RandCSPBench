using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using Random, Statistics, LinearAlgebra
using StableRNGs
using ArgParse

include("src-julia/SAT.jl")
using .SAT: randomcnf, writecnf

function generate_random_cnfs(train_samples, test_samples, traindir, testdir, N, α, k, rng, train_ids, test_ids)
    println("Generating training and testing data with N: $N, alpha: $α...")
    # Filenames only encode (N, M, id), and M = round(N*α) is not injective in α
    # for small N and fine-grained α grids: distinct α's can round to the same M.
    # train_ids/test_ids count samples per M across all α's (instead of resetting
    # id to 1 at every α), so colliding α's accumulate instead of overwriting
    # each other's files.
    for _ in 1:train_samples
        cnf = randomcnf(; N, α, k, rng)
        id = get(train_ids, cnf.M, 0) + 1
        train_ids[cnf.M] = id
        writecnf(joinpath(traindir, "N$(N)_M$(cnf.M)_id$(id).cnf"), cnf)
    end

    for _ in 1:test_samples
        cnf = randomcnf(; N, α, k, rng)
        id = get(test_ids, cnf.M, 0) + 1
        test_ids[cnf.M] = id
        writecnf(joinpath(testdir, "N$(N)_M$(cnf.M)_id$(id).cnf"), cnf)
    end
end

"""
    warn_on_M_collisions(Ns, αs)

round(N*α) is not always injective in α: for small N on a fine-grained α
grid, distinct α's can round to the same M. generate_random_cnfs already
handles this safely (colliding α's accumulate samples at that M instead
of overwriting each other, see its docstring), but it does mean those M
buckets end up with more samples than the rest. Warn so this is visible
rather than silently changing the per-(N,α) sample count.
"""
function warn_on_M_collisions(Ns, αs)
    for N in Ns
        Ms = round.(Int, N .* αs)
        n_dup = length(Ms) - length(unique(Ms))
        if n_dup > 0
            @warn "N=$N: $(n_dup) of the $(length(Ms)) α's in $αs round to an M = round(N*α) already hit by another α; those M's will get extra accumulated samples instead of one sample count per α (see generate_random_cnfs)."
        end
    end
end

function generate_dataset(;
    seed = 17,
    train_samples = 1600,
    test_samples = 400,
    k = 3,
    Ns = [16, 32, 64, 128, 256],
    αs = 3:0.1:5
)
    warn_on_M_collisions(Ns, αs)
    traindir = joinpath(@__DIR__, "$(k)SAT/train")
    testdir = joinpath(@__DIR__, "$(k)SAT/test")
    if isdir(traindir) || isdir(testdir)
        @warn "Directory $traindir or $testdir already exist"
    end
    if train_samples > 0
        mkpath(traindir)
        @info "Writing training set to $(traindir)..."
    end
    if test_samples > 0
        mkpath(testdir)
        @info "Writing test set to $(testdir)..."
    end

    for (i, N) in enumerate(Ns)
        rng = StableRNG(seed + N) # Set respective RNG for each N. Will be used for all αs
        train_ids, test_ids = Dict{Int,Int}(), Dict{Int,Int}()
        for (j, α) in enumerate(αs)
            generate_random_cnfs(train_samples, test_samples, traindir, testdir, N, α, k, rng, train_ids, test_ids)
        end
    end

    println("Data generation procedure complete.")
end

function generate_ood_testset(;
    seed = 17,
    samples = 400,
    k = 3,
    Ns = [512, 1024, 2048, 4096, 8192, 16384],
    αs = 3:0.1:5
)
    warn_on_M_collisions(Ns, αs)
    dir = joinpath(@__DIR__, "$(k)SAT/test_ood")
    @info "Writing out-of-distribution test set to $(dir)..."
    if isdir(dir)
        @warn "Directory already exists!"
    end

    for (i, N) in enumerate(Ns)
        mkpath(dir)
        rng = StableRNG(seed + N) # Set respective RNG for each N. Will be used for all αs
        ids = Dict{Int,Int}()
        for (j, α) in enumerate(αs)
            generate_random_cnfs(0, samples, dir, dir, N, α, k, rng, ids, ids)
        end
    end

    println("Data generation procedure complete.")
end

function (@main)(ARGS)
    s = ArgParseSettings(
        description = "Generate datasets for k-SAT problems on random Erdos-Renyi like CNF formulas.",
        autofix_names = true
    )

    @add_arg_table s begin
        "--test-ood"
            help = "generate out-of-distribution test dataset with larger graphs"
            action = :store_true
    end

    args = parse_args(ARGS, s)

    # Shrink the dataset for CI smoke-testing (GEN_SAT_SMOKE_TEST=true),
    # exercising every code path (including --test-ood) without writing the
    # full-size dataset.
    smoke = get(ENV, "GEN_SAT_SMOKE_TEST", "false") == "true"
    n_train3, n_test3 = smoke ? (2, 2) : (1600, 400)
    n_train4, n_test4 = smoke ? (2, 2) : (800, 200)
    n_ood3, n_ood4 = smoke ? (2, 2) : (400, 200)
    Ns_main = smoke ? [16, 32] : [16, 32, 64, 128, 256]
    Ns_ood = smoke ? [512, 1024] : [512, 1024, 2048, 4096, 8192, 16384]
    αs3 = smoke ? (3.0:0.5:4.0) : (3:0.1:5)
    αs4_train = smoke ? (9.0:0.5:10.0) : (9:0.05:10)
    αs4_test = smoke ? (8.0:0.5:10.0) : (8:0.1:10)

    ## K=3

    ## TRAIN + TEST
    generate_dataset(
        seed = 17,
        train_samples = n_train3,
        test_samples = n_test3,
        k = 3,
        Ns = Ns_main,
        αs = αs3
    )

    ## TEST out-of-distribution
    if args["test_ood"]
        generate_ood_testset(;
            seed = 17,
            samples = n_ood3,
            k = 3,
            Ns = Ns_ood,
            αs = αs3
        )
    end

    ## K=4

    ## TRAIN
    generate_dataset(;
        seed = 17,
        train_samples = n_train4,
        test_samples = n_test4,
        k = 4,
        Ns = Ns_main,
        αs = αs4_train
    )
    rm(joinpath(@__DIR__, "4SAT/test"), recursive=true, force=true) # Drop the test set generated above (different αs), regenerated below

    ## TEST
    generate_dataset(;
        seed = 17,
        train_samples = 0,
        test_samples = n_test4,
        k = 4,
        Ns = Ns_main,
        αs = αs4_test
    )

    ## TEST out-of-distribution
    if args["test_ood"]
        generate_ood_testset(;
            seed = 17,
            samples = n_ood4,
            k = 4,
            Ns = Ns_ood,
            αs = αs4_test
        )
    end
end
