using Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using Random, Statistics, LinearAlgebra
using StableRNGs
using ArgParse

include("src-julia/SAT.jl")
using .SAT: randomcnf, writecnf

function generate_random_cnfs(train_samples, test_samples, traindir, testdir, N, α, k, rng)
    println("Generating training and testing data with N: $N, alpha: $α...")
    for id in 1:train_samples
        cnf = randomcnf(; N, α, k, rng)
        writecnf(joinpath(traindir, "N$(N)_M$(cnf.M)_id$(id).cnf"), cnf)
    end

    for id in 1:test_samples
        cnf = randomcnf(; N, α, k, rng)
        writecnf(joinpath(testdir, "N$(N)_M$(cnf.M)_id$(id).cnf"), cnf)
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
    traindir = joinpath(@__DIR__, "$(k)SAT/train")
    testdir = joinpath(@__DIR__, "$(k)SAT/test")
    if isdir(traindir) || isdir(testdir)
        @warn "Directory $traindir or $testdir already exist"
    end
    if train_samples > 0
        mkpath(traindir)
    end
    if test_samples > 0
        mkpath(testdir)
    end

    for (i, N) in enumerate(Ns)
        rng = StableRNG(seed + N) # Set respective RNG for each N. Will be used for all αs
        for (j, α) in enumerate(αs)
            generate_random_cnfs(train_samples, test_samples, traindir, testdir, N, α, k, rng)
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
    dir = joinpath(@__DIR__, "$(k)SAT/test_ood")
    if isdir(dir)
        @warn "Directory already exists!"
    end

    for (i, N) in enumerate(Ns)
        mkpath(dir)
        rng = StableRNG(seed + N) # Set respective RNG for each N. Will be used for all αs
        for (j, α) in enumerate(αs)
            generate_random_cnfs(0, samples, dir, dir, N, α, k, rng)
        end
    end

    println("Data generation procedure complete.")
end

function (@main)(ARGS)
    s = ArgParseSettings(
        description = "Generate datasets for k-SAT problems on random Erdos-Renyi like CNF formulas."
    )

    @add_arg_table s begin
        "--test-ood"
            help = "generate out-of-distribution test dataset with larger graphs"
            action = :store_true
    end

    args = parse_args(ARGS, s)

    ## K=3

    ## TRAIN + TEST
    generate_dataset(
        seed = 17,
        train_samples = 1600,
        test_samples = 400, 
        k = 3,
        Ns = [16, 32, 64, 128, 256],
        αs = 3:0.1:5
    )

    ## TEST out-of-distribution
    if args["test_ood"]
        generate_ood_testset(;
            seed = 17,
            samples = 400,
            k = 3,
            Ns = [512, 1024, 2048, 4096, 8192, 16384],
            αs = 3:0.1:5
        )
    end

    ## K=4

    ## TRAIN
    generate_dataset(;
        seed = 17,
        train_samples = 800,
        test_samples = 200, 
        k = 4,
        Ns = [16, 32, 64, 128, 256],
        αs = 9:0.05:10
    )
    rm(joinpath(@__DIR__, "4SAT/train"), force=true) # Remove test set generated for retrocompatibility
    
    ## TEST
    generate_dataset(;
        seed = 17,
        train_samples = 0,
        test_samples = 200,
        k = 4,
        Ns = [16, 32, 64, 128, 256],
        αs = 8:0.1:10
    )
end
