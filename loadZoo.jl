using CSV
using DataFrames
using Plots
using DataSets
using StatsPlots
using Tar

file = open(Blob, dataset("zoo")) do blob
    open(IO, blob) do io
        buf = read(blob)
        full_file = CSV.read(buf, DataFrame)
        groupedbar(full_file.Month,
                   [full_file.Max_age full_file.Min_age],
                   labels = ["Max_age" "Min_age"],
                   title = "Max/min age of purchased animals",
                   size = (925, 450))
        results_dir = joinpath(@__DIR__, "results")
        mkdir(results_dir)
        savefig(joinpath(results_dir, "animalsAge.pdf"))
        scatter(full_file.Animal,
                full_file.Count,
                labels = "total number",
                title = "Total number of purchased animals",
                size=(925, 450))
        savefig(joinpath(results_dir, "animalsCount.pdf"))
        tarball = Tar.create(results_dir)
        ENV["RESULTS_FILE"] = tarball
    end
end
