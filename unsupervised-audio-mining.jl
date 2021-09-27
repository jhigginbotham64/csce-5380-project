### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 4ba0aa6e-1fc5-11ec-1bc9-8790f201354a
begin
	using WAV
	using CSV
	using DelimitedFiles
	using DataFrames
	using GoogleDrive
	using DataSets
	using Flux
	using Plots
	using StatsPlots
	using OhMyREPL
	using PlutoUI
end

# ╔═╡ 1895ece2-fb9d-4237-af5f-10543c0a28d9
project = DataSets.load_project(Dict(
"data_config_version"=>0,
"datasets"=>[Dict(
	"name"=>"chime_home",
	"uuid"=>"73b60068-bf34-11eb-17e9-5fa4ccb60cd2", # UUID generated externally, no need to change
	"description"=>"4-second audio chunks at different sample rates with labels and other metadata",
	"storage"=>Dict(
		"driver"=>"FileSystem",
		"type"=>"BlobTree",
		"path"=>joinpath(homedir(), ".julia", "datadeps", "chime_home")
		)
	)]
))

# ╔═╡ 6ea126c1-f0cf-458d-bd0a-4b13c9b18a01
chime_home = open(BlobTree, dataset(project, "chime_home"))

# ╔═╡ a41923c9-e6f5-4c5a-880d-f065d20588ff
function print_data_file(s)
    open(String, chime_home[s]) do data
        print(data)
    end
end

# ╔═╡ a205965f-b43b-4bb8-9a42-1ec190f3b9ad
print_data_file("README") # prints to whichever terminal Pluto was initially run from

# ╔═╡ b21ff8cc-d39b-48dd-b6af-ced0cca5608a
begin
	# read all chunk grouping CSV files into 
	# DataFrames with appropriate headers
	chunk_headers=["id", "name"];
	dev_chunks_raw = CSV.read(
	    IOBuffer(open(String, chime_home["development_chunks_raw.csv"])), 
	    DataFrame; header=chunk_headers);
	dev_chunks_refined = CSV.read(
	    IOBuffer(open(String, chime_home["development_chunks_refined.csv"])), 
	    DataFrame; header=chunk_headers);
	eval_chunks_raw = CSV.read(
	    IOBuffer(open(String, chime_home["evaluation_chunks_raw.csv"])), 
	    DataFrame; header=chunk_headers);
	eval_chunks_refined = CSV.read(
	    IOBuffer(open(String, chime_home["evaluation_chunks_refined.csv"])), 
	    DataFrame; header=chunk_headers);
	dev_chunks_refined_folds = CSV.read(
	    IOBuffer(open(String, 	
				chime_home["development_chunks_refined_crossval_dcase2016.csv"])), 
	    DataFrame; header=vcat(chunk_headers, ["fold"]));
	chunks = chime_home["chunks"];
end

# ╔═╡ Cell order:
# ╠═4ba0aa6e-1fc5-11ec-1bc9-8790f201354a
# ╠═1895ece2-fb9d-4237-af5f-10543c0a28d9
# ╠═6ea126c1-f0cf-458d-bd0a-4b13c9b18a01
# ╠═a41923c9-e6f5-4c5a-880d-f065d20588ff
# ╠═a205965f-b43b-4bb8-9a42-1ec190f3b9ad
# ╠═b21ff8cc-d39b-48dd-b6af-ced0cca5608a
