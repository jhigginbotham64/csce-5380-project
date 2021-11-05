### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ 4ba0aa6e-1fc5-11ec-1bc9-8790f201354a
begin
	using WAV
	using CSV
	using DataFrames
	using DataSets
	using Flux
	using Flux: @epochs
	using OhMyREPL
	using DotEnv
	using MFCC
	using CUDA
	using Random
end

# ╔═╡ b0bd58fb-3758-4909-a337-020ffb1752e4
md"""
	next tasks:
	- need to code:
		- MBK (done)
		- MFCC (done)
		- multi-class encoding (done)
		- background noise aware training (clear if understood as adaptation,
		i.e. augmentation also occurs during prediction, i.e. not during data loading)
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
	solve in this domain. besides memory consumption, which can be solved by primarily
	preserving analytics data and only preserving audio chunks up to a certain
	threshold per class. bada-bing bada-boom, a self-hosted audio surveillance app.
	...i'll need to polish that and maybe add some visualizations if i want to use
	it in my paper, but like...i don't have to use it in my paper, i can just use it
	for myself. however, it does justify focusing on the multiclass problem over and
	above the single-class problem, i.e. properly distinguishing audio tagging from
	AED.
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

# ╔═╡ b71a1abd-0b85-471e-9347-60638916b99b
function get_df_val(df, key)
   val = String(df[df.key .== key, :].val[1])
   if key in ["segmentname", "chunkname", 
				   "annotation_a1", "annotation_a2",
				   "annotation_a3", "majorityvote"] return val
   elseif key in ["chunknumber", "framestart"] return parse(Int, val)
   elseif key in ["session_a1", "session_a2", 
				   "session_a3"] return parse(Float64, val)
   end
end

# ╔═╡ 52518e6f-d42b-45bf-a5be-b02e899c88e5
function preprocess_chunk(chunkname)
	c = CSV.read(
	   IOBuffer(open(String, chime_home["chunks"][chunkname * ".csv"])), 
	   DataFrame; header=["key","val"])
	c = Dict(key => get_df_val(c, key) for key in c.key)
	c["c"] = 'c' in c["majorityvote"] ? 1 : 0
	c["m"] = 'm' in c["majorityvote"] ? 1 : 0
	c["f"] = 'f' in c["majorityvote"] ? 1 : 0
	c["v"] = 'v' in c["majorityvote"] ? 1 : 0
	c["p"] = 'p' in c["majorityvote"] ? 1 : 0
	c["b"] = 'b' in c["majorityvote"] ? 1 : 0
	c["o"] = 'o' in c["majorityvote"] ? 1 : 0
	c["labels"] = [c["c"], c["m"], c["f"], c["v"], c["p"], c["b"], c["o"]]
	data, freq, _ = wavread(
		joinpath(
			project.datasets["chime_home"].storage["path"], "chunks", 
			chunkname * ".16kHz.wav"))
	c["data"] = data
	c["freq"] = freq
	# wintime, steptime, and numcep values are all taken from the paper
	datamfcc = mfcc(
		reshape(data, 64000), freq; wintime=0.02, steptime=0.01, numcep=24)
	c["mfcc"] = znorm(datamfcc[1])
	c["mbk"] = znorm(datamfcc[2][:, 1:40]) # 40 (nummbk) is also from the paper
	return c
end

# ╔═╡ e5fe10f1-b9fd-4802-a0da-911c065d2967
function get_chunk_df(df)
	newdf = DataFrame(
		segmentname = String[],
		chunknumber = Int[],
		framestart = Int[],
		chunkname = String[],
		annotation_a1 = String[],
		session_a1 = AbstractFloat[],
		annotation_a2 = String[],
		session_a2 = AbstractFloat[],
		annotation_a3 = String[],
		session_a3 = AbstractFloat[],
		majorityvote = String[],
		c = Int[],
		m = Int[],
		f = Int[],
		v = Int[],
		p = Int[],
		b = Int[],
		o = Int[],
		labels = Vector{Int}[],
		data = Matrix{AbstractFloat}[],
		freq = AbstractFloat[],
		mfcc = Matrix{AbstractFloat}[],
		mbk = Matrix{AbstractFloat}[]
	)

	for chunkname in df.chunkname
		push!(newdf, preprocess_chunk(chunkname))
	end

	return newdf
end

# ╔═╡ 127dcb77-63f3-46fd-ad7b-7a186f37b81a
# get chime home csv dataframe
function get_ch_csv_df(fname; header=["id", "chunkname"])
	return CSV.read(
		IOBuffer(open(String, chime_home[fname])), 
		DataFrame; header=header)
end

# ╔═╡ 6fb6eebf-a490-4117-afb0-290f465e210c
get_chunk_set(fname) = get_chunk_df(get_ch_csv_df(fname))

# ╔═╡ d4dbe98f-7919-4d73-b4c9-2524359e7292
function chunk_set_loader_batched(fname; feature, batchsize=100)
	df = get_chunk_set(fname)
	return Flux.Data.DataLoader((data=df[:, feature], 
		labels=df[:, "labels"]), batchsize=batchsize, shuffle=true)
end

# ╔═╡ 71b3dfa1-afe7-4ce8-8724-b552267ac6aa
function chunk_set_loader_unbatched(fname; feature)
	df = get_chunk_set(fname)
	return zip(df[:, feature], df[:, "labels"])
end

# ╔═╡ 9591981b-63ea-4edf-9cef-2760963ff3ed
function DNN(in; 
	dropout1 = 0.1,
	dropout2 = 0.2,
	dense1 = 1000,
	dense2 = 500
)
	return Chain(
		vec, # flatten input
		# Dropout(dropout1),
		Dense(in, dense1, relu),
		Dropout(dropout2),
		Dense(dense1, dense2, relu),
		Dropout(dropout2),
		Dense(dense2, 7, sigmoid), # 7 = number of labels
	)
end

# ╔═╡ e3135a53-14e0-46f6-8a5a-3f0450dc4014
begin
	feature = "mfcc" # which column to use as input
	# data loaders
	# trn = chunk_set_loader_batched(
	# 	"development_chunks_refined.csv"; feature=feature) 
	# tst = chunk_set_loader_batched(
	# 	"evaluation_chunks_refined.csv"; feature=feature)
	trn = chunk_set_loader_unbatched(
		"development_chunks_refined.csv"; feature=feature) 
	tst = chunk_set_loader_unbatched(
		"evaluation_chunks_refined.csv"; feature=feature)
end

# ╔═╡ 6a5c11a3-a8d4-4491-9637-8472cc076861
begin
	# parameters
	η = 0.005 # learning rate
	ρ = 0.9 # momentum
	seed = 0xDEADBEEF # random seed
	seed > 0 && Random.seed!(seed)
	nepochs = 1 # training iterations
	cuda = true # use gpu if available
	loss_func = Flux.Losses.binarycrossentropy

	# set device for later
	if cuda && CUDA.has_cuda()
		device = gpu
		# whether we want to allow avoiding the gpu or not
		CUDA.allowscalar(false)
	else
		device = cpu
	end
end

# ╔═╡ d698686b-ee9e-4a95-91f4-f03e9c59df26
begin
	dnn = DNN(399 * 24) |> device
	dp = params(dnn)
	
	function loss(x, y)
		loss_func(dnn(x |> device), y |> device)
		# println("exploring: ", string(typeof(d)))
		# for d_ in d
		# 	println(string(typeof(d_)))
		# 	println(string(length(d_)))
		# end
	end
	
	# function cb()
	# 	println("tst loss: ", string(loss(first(tst)...)))
	# end
	
	# @epochs nepochs Flux.train!(loss, dp, trn, Flux.Momentum(η, ρ); cb=cb)
	@epochs nepochs Flux.train!(loss, dp, trn, Flux.Momentum(η, ρ))
end

# ╔═╡ e8d57d33-a224-4150-9308-69455db18465
md"""
	ok here's the DNN code, all the other dnn's are the same
	except that they use binary_crossentropy instead of mse
	(this is the mfcc dnn):

	###build model by keras
	model = Sequential()

	#model.add(Flatten(input_shape=(agg_num,fea_dim)))
	model.add(Dropout(0.1,input_shape=(agg_num*fea_dim+fea_dim,)))
	#model.add(Dropout(0.1,input_shape=(agg_num*fea_dim,)))

	model.add(Dense(1000,input_dim=agg_num*fea_dim))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(500))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(n_out))
	model.add(Activation('sigmoid'))

	model.summary()

	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	#model.compile(loss='mse', optimizer='adam') ### sth wrong here
	sgd = SGD(lr=0.005, decay=0, momentum=0.9)
	#model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	model.compile(loss='mse', optimizer=sgd)

	dump_fd=cfg.scrap_fd+'/Md/dnn_mfc24_fold1_fr91_bcCOST_keras_weights.{epoch:02d}-{val_loss:.2f}.hdf5'

	eachmodel=ModelCheckpoint(dump_fd,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto')      

	model.fit(tr_X, tr_y, batch_size=100, nb_epoch=51,
				  verbose=1, validation_data=(te_X, te_y), callbacks=[eachmodel]) #, callbacks=[best_model])
	#score = model.evaluate(te_X, te_y, show_accuracy=True, verbose=0)
	#print('Test score:', score[0])
	#print('Test accuracy:', score[1])

	...and here's the aDAE:

	###build model by keras
	input_audio=Input(shape=(agg_num*fea_dim,))
	encoded = Dropout(0.1)(input_audio)
	encoded = Dense(500,activation='relu')(encoded)
	encoded = Dense(50,activation='relu')(encoded)

	decoded = Dense(500,activation='relu')(encoded)
	#decoded = Dense(fea_dim*agg_num,activation='linear')(decoded)
	decoded = Dense(fea_dim,activation='linear')(decoded)

	autoencoder=Model(input=input_audio,output=decoded)

	autoencoder.summary()

	sgd = SGD(lr=0.01, decay=0, momentum=0.9)
	autoencoder.compile(optimizer=sgd,loss='mse')

	dump_fd=cfg.scrap_fd+'/Md/dae_keras_Relu50_1outFr_7inFr_dp0.1_weights.{epoch:02d}-{val_loss:.2f}.hdf5'

	eachmodel=ModelCheckpoint(dump_fd,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto') 

	autoencoder.fit(tr_X,tr_y,nb_epoch=100,batch_size=100,shuffle=True,validation_data=(te_X,te_y), callbacks=[eachmodel])
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataSets = "c9661210-8a83-48f0-b833-72e62abce419"
DotEnv = "4dc1fcf4-5e3b-5448-94ab-0c38ec0385c1"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
MFCC = "ca7b5df7-6146-5dcc-89ec-36256279a339"
OhMyREPL = "5fb14364-9ced-5910-84b2-373655c76a03"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
WAV = "8149f6b0-98f6-5db9-b78f-408fbbb8ef88"

[compat]
CSV = "~0.9.9"
CUDA = "~2.6.3"
DataFrames = "~1.2.2"
DataSets = "~0.2.5"
DotEnv = "~0.3.1"
Flux = "~0.12.1"
MFCC = "~0.3.1"
OhMyREPL = "~0.5.10"
WAV = "~1.1.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[BFloat16s]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "4af69e205efc343068dc8722b8dfec1ade89254a"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.1.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Blosc]]
deps = ["Blosc_jll"]
git-tree-sha1 = "84cf7d0f8fd46ca6f1b3e0305b4b4a37afe50fd6"
uuid = "a74b3585-a348-5f62-a45c-50e91977d574"
version = "0.7.0"

[[Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "e747dac84f39c62aff6956651ec359686490134e"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.0+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "c0a735698d1a0a388c5c7ae9c7fb3da72fd5424e"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.9"

[[CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "DataStructures", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "MacroTools", "Memoize", "NNlib", "Printf", "Random", "Reexport", "Requires", "SparseArrays", "Statistics", "TimerOutputs"]
git-tree-sha1 = "6893a46f357eabd44ce0fc1f9a264120a1a3a732"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "2.6.3"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3ae0487c35784c859c485383541beaa0c1560d3d"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "0.8.25"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f53ca8d41e4753c41cdafa6ec5f7ce914b34be54"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.13"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DSP]]
deps = ["FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "2a63cb5fc0e8c1f0f139475ef94228c7441dc7d0"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.6.10"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataSets]]
deps = ["AbstractTrees", "Base64", "Markdown", "REPL", "ReplMaker", "ResourceContexts", "SHA", "TOML", "UUIDs"]
git-tree-sha1 = "d35089f32dfd665cf45e2b408be6e846c4191727"
uuid = "c9661210-8a83-48f0-b833-72e62abce419"
version = "0.2.5"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "7220bc21c33e990c14f4a9a319b1d242ebc5b269"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.1"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[DotEnv]]
git-tree-sha1 = "d48ae0052378d697f8caf0855c4df2c54a97e580"
uuid = "4dc1fcf4-5e3b-5448-94ab-0c38ec0385c1"
version = "0.3.1"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "3c041d2ac0a52a12a27af2782b34900d9c3ee68c"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.1"

[[FilePathsBase]]
deps = ["Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "7fb0eaac190a7a68a56d2407a6beff1142daf844"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.12"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Flux]]
deps = ["AbstractTrees", "Adapt", "CUDA", "CodecZlib", "Colors", "DelimitedFiles", "Functors", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "Pkg", "Printf", "Random", "Reexport", "SHA", "Statistics", "StatsBase", "Test", "ZipFile", "Zygote"]
git-tree-sha1 = "287705d01ab510afe075b0165a159b9e5a4bf082"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.12.1"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "63777916efbcb0ab6173d09a658fb7f2783de485"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.21"

[[Functors]]
git-tree-sha1 = "e4768c3b7f597d5a352afa09874d16e3c3f6ead2"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.7"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GPUArrays]]
deps = ["AbstractFFTs", "Adapt", "LinearAlgebra", "Printf", "Random", "Serialization", "Statistics"]
git-tree-sha1 = "df5b8569904c5c10e84c640984cfff054b18c086"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "6.4.1"

[[GPUCompiler]]
deps = ["DataStructures", "ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "Serialization", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "ef2839b063e158672583b9c09d2cf4876a8d3d55"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.10.0"

[[HDF5]]
deps = ["Blosc", "HDF5_jll", "Libdl", "Mmap", "Random"]
git-tree-sha1 = "0b812e7872e2199a5a04944f486b4048944f1ed8"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.13.7"

[[HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "bc9c3d43ffd4d8988bfa372b86d4bdbd26860e95"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.10.5+7"

[[IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "95215cd0076a150ef46ff7928892bc341864c73c"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.3"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "19cb49649f8c41de7fea32d089d37de917b553da"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.0.1"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Intervals]]
deps = ["Dates", "Printf", "RecipesBase", "Serialization", "TimeZones"]
git-tree-sha1 = "323a38ed1952d30586d0fe03412cde9399d3618b"
uuid = "d8418881-c3e1-53bb-8760-2df7ec849ed5"
version = "1.5.0"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a8d93f02e9db5428ccb73336a54c746e3e87edd3"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.4"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[LLVM]]
deps = ["CEnum", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "f57ac3fd2045b50d3db081663837ac5b4096947e"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "3.9.0"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[MFCC]]
deps = ["DSP", "Distributed", "FileIO", "HDF5", "SpecialFunctions", "Statistics", "WAV"]
git-tree-sha1 = "e8d6bb66e00f85ea7ba7f244da3b097d80825b3b"
uuid = "ca7b5df7-6146-5dcc-89ec-36256279a339"
version = "0.3.1"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "29714d0a7a8083bba8427a4fbfb00a540c681ce7"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "5203a4532ad28c44f82c76634ad621d7c90abcbd"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.29"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[OhMyREPL]]
deps = ["Crayons", "JLFzf", "Markdown", "Pkg", "Printf", "REPL", "Tokenize"]
git-tree-sha1 = "646c8cf453f25f12115ee09d57ca192c9af00618"
uuid = "5fb14364-9ced-5910-84b2-373655c76a03"
version = "0.5.10"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "f19e978f81eca5fd7620650d7dbea58f825802ee"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.0"

[[Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[Polynomials]]
deps = ["Intervals", "LinearAlgebra", "OffsetArrays", "RecipesBase"]
git-tree-sha1 = "0b15f3597b01eb76764dd03c3c23d6679a3c32c8"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "1.2.1"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "d940010be611ee9d67064fe559edbb305f8cc0eb"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.3"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[ReplMaker]]
deps = ["REPL", "Unicode"]
git-tree-sha1 = "76098218397ec93b925b70ce355144d539b1a8b4"
uuid = "b873ce64-0db9-51f5-a568-4457d8e49576"
version = "0.2.5"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[ResourceContexts]]
git-tree-sha1 = "a51e18013ef68dfdc3dcdfc7ba5f6659dbdc7cbf"
uuid = "8d208092-d35c-4dd3-a0d7-8325f9cce6b4"
version = "0.1.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "f45b34656397a1f6e729901dc9ef679610bd12b5"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.8"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["OpenSpecFun_jll"]
git-tree-sha1 = "d8d8b8a9f4119829410ecd706da4cc8594a1e020"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "0.10.3"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimeZones]]
deps = ["Dates", "Downloads", "InlineStrings", "LazyArtifacts", "Mocking", "Pkg", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "b4c6460412b1db0b4f1679ab2d5ef72568a14a57"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.6.1"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "7cb456f358e8f9d102a8b25e8dfedf58fa5689bc"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.13"

[[Tokenize]]
git-tree-sha1 = "0952c9cee34988092d73a5708780b3917166a0dd"
uuid = "0796e94c-ce3b-5d07-9a54-7f471281c624"
version = "0.5.21"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[WAV]]
deps = ["Base64", "FileIO", "Libdl", "Logging"]
git-tree-sha1 = "1d5dc6568ab6b2846efd10cc4d070bb6be73a6b8"
uuid = "8149f6b0-98f6-5db9-b78f-408fbbb8ef88"
version = "1.1.1"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "IRTools", "InteractiveUtils", "LinearAlgebra", "MacroTools", "NaNMath", "Random", "Requires", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "8b634fdb4c3c63f2ceaa2559a008da4f405af6b3"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.17"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "547d2167e376a423ad3c161c0aba34308cc897e7"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.27.2+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═b0bd58fb-3758-4909-a337-020ffb1752e4
# ╠═4ba0aa6e-1fc5-11ec-1bc9-8790f201354a
# ╠═bb8d065b-1495-4390-a782-94571ba40275
# ╠═1895ece2-fb9d-4237-af5f-10543c0a28d9
# ╠═6ea126c1-f0cf-458d-bd0a-4b13c9b18a01
# ╠═a41923c9-e6f5-4c5a-880d-f065d20588ff
# ╠═b71a1abd-0b85-471e-9347-60638916b99b
# ╠═52518e6f-d42b-45bf-a5be-b02e899c88e5
# ╠═e5fe10f1-b9fd-4802-a0da-911c065d2967
# ╠═127dcb77-63f3-46fd-ad7b-7a186f37b81a
# ╠═6fb6eebf-a490-4117-afb0-290f465e210c
# ╠═d4dbe98f-7919-4d73-b4c9-2524359e7292
# ╠═71b3dfa1-afe7-4ce8-8724-b552267ac6aa
# ╠═9591981b-63ea-4edf-9cef-2760963ff3ed
# ╠═e3135a53-14e0-46f6-8a5a-3f0450dc4014
# ╠═6a5c11a3-a8d4-4491-9637-8472cc076861
# ╠═d698686b-ee9e-4a95-91f4-f03e9c59df26
# ╠═e8d57d33-a224-4150-9308-69455db18465
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
