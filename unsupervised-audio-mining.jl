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

# ╔═╡ b0bd58fb-3758-4909-a337-020ffb1752e4
md"""
	next tasks:
	- need to code:
		- custom MBK (clear)
		- custom MFCC (clear)
		- multi-class encoding (clear)
		- background noise aware training (clear if understood as adaptation,
		i.e. augmentation also occurs during prediction)
		- define data loaders (clear)
		- DNN baseline using raw audio data (clear, altho departs from paper)
		- MFCC-DNN (clear)
		- MBK-DNN (clear)
		- aDAE (train on MBK, reconstruct middle frame from 7)
		- get features from trained aDAE (custom training and 
		testing loop, maybe need to freeze certain parameters)
		- aDAE-DNN (clear)
		- class precision (clear)
		- class recall (clear)
		- class F1 (clear)
		- confusion matrix (clear)
		- EER (clear)
	- investigate:
		- exploring techniques from other papers

	notes on other papers:
	- paper 2 does the same thing as paper 1, only the problem it tackles is
	single-class classification rather than multiclass, and it uses a ConvLSTM
	autoencoder rather than a DAE. it also explores how to use the learned
	features for simpler classification as well as for clustering, focusing
	deeply on the quality of the learned features, and presents a novel training
	technique designed to optimize for intraclass similarity rather than for 
	the stability of cluster centers. "could be expanded to perform clustering
	itself", but no exploration is made in that direction. if i was to write code
	based on it, it would be...tricky. unsure how i would adapt it from single
	class to multiclass, altho it would be interesting to try. i suppose the fact
	that the unsupervised aspect takes the form of MBK frame prediction makes the
	single/multiclass distinction somewhat irrelevant tho, at least as far as the
	most interesting elements of the paper are concerned, and it would be interesting
	to see how it performs on noisy data compared to the aDAE, especially if the
	same background noise augmentation was applied. would need to explore whether
	dropout would be beneficial in the new architecture. would need to adapt the
	network parameters to the new data anyway, and if that was done then the same
	dropout could be used, problem solved, although a comparison would still need
	to be done. would be super interesting to do paper 2's classification and 
	clustering comparisons using chime_home and the aDAE, really pitting it against
	the ConvLSTM, and *that's* the part where single/multi-class adapatation becomes
	a true barrier. but i'd just need to use algorithms and conditions other than
	what paper 2 uses, and anyhow the ConvLSTM has the same problem with chime_home.
	hooray for learning! make me also wonder if their novel training technique could
	be applied to the aDAE with any benefit for clustering or classification.
	...i see now that "classification performance" is just multiclass performance
	against chime_home's evaluation set, as in paper 1, whether with DNN or without. 
	the real kicker is how to define a clustering problem for chime_home, since
	clusters overlap. but i can leave that as a problem for another time, even if
	it hints at what for me is the true central problem (event extraction).
	- paper 3 classifies audio snippets using a CNN based on VGGNet. no separate
	feature learning is done, and reasons are given for this. they distinguish
	their approach from previous frame-feature-based approaches by explaining their
	"large input field", which makes it basically sound like they feed the snippets
	to their network raw. lovely. they describe a regularization method that they
	used, as well as a data augmentation method. they also use multiple instance
	learning to handle noise in their custom data set. this all represents a 
	fundamentally different approach to AED from those of papers 1 and 2, one that
	might withstand comparison against them if adapted to multiclass on chime_home.
	i think that's what i want to do. that's what i'm going to do.

	other notes:
	- i really need to bear down on my ideas for a futuristic audio surveillance
	system if i want to paint a picture of it rather than just compare some networks
	on a classification task. and i need to figure it out one way or another, because
	it will determine the direction of this project and perhaps future personal 
	efforts. and i want to get it out of my head too, you know? say you have recording
	equipment, storage devices, and a GPU machine capable of training and deploying
	deep learning models. say you start with no data. take it all as a multiclass
	problem, splitting the stream into windows and classifying each window, even
	ones that overlap (it's important that they overlap). the larger the window,
	the more background processing is required, the less certainty you'll have about
	finding a particular event in particular time frame, but the more overlapping
	segments you'll have to work with. say your window size is 4, as with chime_home.
	you start with no data. record an hour. label parts of the recording with known
	and unknown events, allowing events to overlap. split the recording into
	overlapping windows, use whichever events are present in each as your labels,
	train a multiclass classification algorithm. make sure one of the classes is
	"other/unknown". more recordings, more labelings, more data, more training.
	here's the feedback loop: deploy, and let your online system capture events and
	label them itself. then come back to examine and process the data, paying careful
	attention to mislabeled areas, and adding new ones as appropriate to break up the
	"other/unknown" category. relabel, retrain, redeploy, rinse, lather, repeat.
	parameters: number of windows, size of windows, and size of "minimum overlap
	frame". plug and play with whatever classification algorithms you like. optimize
	for response time, certainty of labels, memory consumption, level of user
	interaction required, etc. the deployed version is capable of sending sound event
	notifications in real time as well as performing event visualization and 
	analytics. the training feedback loop allows improvement in the underlying 
	classification algorithm, as well as for the identification of unknown events.
	the key insight is that multiclass algorithms can accommodate stream analysis 
	using a voting system, which can provide start and end times for individual
	events via overlapping windows without the need to manually extract them or even
	identify them individually, which would otherwise be the hardest problems to
	solve in this domain.
	...i'll need to polish that and maybe add some visualizations if i want to use
	it in my paper, but like...i don't have to use it in my paper, i can just use it
	for myself. however, it does justify focusing on the multiclass problem over and
	above the single-class problem.
"""

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

# ╔═╡ Cell order:
# ╠═b0bd58fb-3758-4909-a337-020ffb1752e4
# ╠═4ba0aa6e-1fc5-11ec-1bc9-8790f201354a
# ╠═bb8d065b-1495-4390-a782-94571ba40275
# ╠═1895ece2-fb9d-4237-af5f-10543c0a28d9
# ╠═6ea126c1-f0cf-458d-bd0a-4b13c9b18a01
# ╠═a41923c9-e6f5-4c5a-880d-f065d20588ff
# ╠═b21ff8cc-d39b-48dd-b6af-ced0cca5608a
