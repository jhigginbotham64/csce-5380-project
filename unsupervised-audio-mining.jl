### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 4ba0aa6e-1fc5-11ec-1bc9-8790f201354a
begin
	using WAV
	using CSV
	using DataFrames
	using DataSets
	using Flux
	using OhMyREPL
	using DotEnv
end

# ╔═╡ bb8d065b-1495-4390-a782-94571ba40275
cfg = DotEnv.config()

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
		"path"=>ENV["chime_home"] # on-disk location of extracted chime_home dataset
		)
	)]
))

# ╔═╡ 6ea126c1-f0cf-458d-bd0a-4b13c9b18a01
chime_home = open(BlobTree, dataset(project, "chime_home"))

# ╔═╡ a41923c9-e6f5-4c5a-880d-f065d20588ff
function print_data_file(s)
    open(String, chime_home[s]) do data
        print(data) # prints to whichever terminal Pluto was initially run from
    end
end

# ╔═╡ b21ff8cc-d39b-48dd-b6af-ced0cca5608a
begin
	chunks_dir = chime_home["chunks"];
	
	function getdfval(df, key)
	   val = String(df[df.key .== key, :].val[1])
	   if key in ["segmentname", "chunkname", 
					   "annotation_a1", "annotation_a2",
					   "annotation_a3", "majorityvote"] return val
	   elseif key in ["chunknumber", "framestart"] return parse(Int, val)
	   elseif key in ["session_a1", "session_a2", 
					   "session_a3"] return parse(Float64, val)
	   end
	end
	
	function preprocess_chunk(chunkname)
		c = CSV.read(
		   IOBuffer(open(String, chunks_dir[chunkname * ".csv"])), 
		   DataFrame; header=["key","val"])
		c = Dict(key => getdfval(c, key) for key in c.key)
		# because right now i don't know the first thing about what this stuff means
		wav1, wav2, wav3, wav4 = wavread(
			joinpath(
				project.datasets["chime_home"].storage["path"], "chunks", 
				chunkname * ".16kHz.wav"))
		c["wav1"] = wav1
		c["wav2"] = wav2
		c["wav3"] = wav3
		c["wav4"] = wav4[1] # vector is only length 1, so unpack it here
		return c
	end
	
	function getchunkdf(df)
		newdf = DataFrame(
			segmentname = String[],
			chunknumber = Int[],
			framestart = Int[],
			annotation_a1 = String[],
			session_a1 = AbstractFloat[],
			annotation_a2 = String[],
			session_a2 = AbstractFloat[],
			annotation_a3 = String[],
			session_a3 = AbstractFloat[],
			majorityvote = String[],
			chunkname = String[],
			wav1 = Matrix{AbstractFloat}[],
			wav2 = AbstractFloat[],
			wav3 = Int[],
			wav4 = WAVChunk[],
		)

		for chunkname in df.chunkname
			push!(newdf, preprocess_chunk(chunkname))
		end
		
		return newdf
	end
	
	chunk_headers=["id", "chunkname"];
	eval_chunks = getchunkdf(CSV.read(
		IOBuffer(open(String, chime_home["evaluation_chunks_refined.csv"])), 
		DataFrame; header=chunk_headers));
	dev_chunks = getchunkdf(CSV.read(
		IOBuffer(open(String, chime_home["development_chunks_refined.csv"])), 
		DataFrame; header=chunk_headers));
end

# ╔═╡ b0bd58fb-3758-4909-a337-020ffb1752e4
"""
	next tasks:
	- need to code:
		- custom MBK (clear)
		- custom MFCC (clear)
		- multi-class encoding (clear)
		- DNN baseline using raw audio data (clear, altho departs from paper)
		- MFCC-DNN (clear)
		- MBK-DNN (clear)
		- aDAE -> paper
		- get features from trained aDAE -> Flux docs and paper
		- aDAE-DNN (clear)
		- class precision (clear)
		- class recall (clear)
		- class F1 (clear)
		- confusion matrix (clear)
		- EER (clear)
"""

# ╔═╡ Cell order:
# ╠═4ba0aa6e-1fc5-11ec-1bc9-8790f201354a
# ╠═bb8d065b-1495-4390-a782-94571ba40275
# ╠═1895ece2-fb9d-4237-af5f-10543c0a28d9
# ╠═6ea126c1-f0cf-458d-bd0a-4b13c9b18a01
# ╠═a41923c9-e6f5-4c5a-880d-f065d20588ff
# ╠═b21ff8cc-d39b-48dd-b6af-ced0cca5608a
# ╠═b0bd58fb-3758-4909-a337-020ffb1752e4
