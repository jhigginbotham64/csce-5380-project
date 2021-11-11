### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ bd650de6-7ee1-454f-8d53-270b43f70770
begin
	import Pkg
	Pkg.activate(".")
end

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
function chunk_set_loader_batched(df; feature, batchsize=32)
	return Flux.Data.DataLoader((data=[vec(f) for f in df[:, feature]], 
		labels=[vec(l) for l in df[:, "labels"]]), batchsize=batchsize, shuffle=true)
end

# ╔═╡ 71b3dfa1-afe7-4ce8-8724-b552267ac6aa
function chunk_set_loader_unbatched(df; feature)
	return zip([vec(f) for f in df[:, feature]], [vec(l) for l in df[:, "labels"]])
end

# ╔═╡ 6a5c11a3-a8d4-4491-9637-8472cc076861
begin
	seed = 0xDEADBEEF # random seed
	seed > 0 && Random.seed!(seed)
	nepochs = 100 # training iterations
	cuda = true # use gpu if available

	# set device for later
	if cuda && CUDA.has_cuda()
		device = gpu
		# whether we want to allow avoiding the gpu or not
		CUDA.allowscalar(false)
	else
		device = cpu
	end

	# unbatched device loss, i.e. calculate loss on device
	# where the outputs and labels are not batched
	udl(lss, mdl) = (x, y) -> lss(mdl(x |> device), y |> device)
	
	# batched device loss, based on the docs:
	# https://fluxml.ai/Flux.jl/stable/performance/#Evaluate-batches-as-Matrices-of-features
	function bdl(lss, mdl)
		function (d)
			xs = reduce(hcat, d.data) |> device
			ys = reduce(hcat, d.labels) |> device
			yhats = mdl(xs) |> device
			return sum(lss.(yhats, ys))
		end
	end
	
	# opt = () -> Flux.Optimise.Momentum(0.005, 0.9)
	opt = () -> Flux.Optimise.ADAM()
	
	throttle_sec = 15
	throttled = (f) -> Flux.throttle(f, throttle_sec)

	cb(lss) = throttled(
			function ()
				println("tst loss: ", string(lss), " ", string(lss()))
			end
		)

	tst_loss_unbatched = (tst, lss) -> () -> sum([lss(d...) for d in tst])
	tst_loss_batched = (tst, lss) -> () -> sum([lss(d) for d in tst])
end

# ╔═╡ 9591981b-63ea-4edf-9cef-2760963ff3ed
function DNN(in; 
	dropout1 = 0.1, # default from paper
	dropout2 = 0.2,
	dense1 = 1000,
	dense2 = 500
)
	return Chain(
		Dropout(dropout1),
		Dense(in, dense1, relu),
		Dropout(dropout2),
		Dense(dense1, dense2, relu),
		Dropout(dropout2),
		Dense(dense2, 7, sigmoid), # 7 = number of labels
	)
end

# ╔═╡ c9e50684-8e04-4481-a2fe-8612c1c6e8fd
function aDAE(in; # asynchronous denoising autoencoder
	frames = 7, # defaults from paper
	dropout = 0.1,
	dense = 500,
	bottleneck = 50
)
	return Chain(
		# encoder
		Dropout(dropout),
		Dense(in, dense, relu),
		Dense(dense, bottleneck, relu),
		# decoder
		Dense(bottleneck, dense, relu),
		Dense(dense, Int(in / frames)) # default activation is identity/linear
	)
end

# ╔═╡ 64856590-dd0e-4708-96b2-e776bd46796a
# takes a zipped set of features and labels
# and creates a new set where the labels are
# the middle frames of the features, which
# of course assumes an odd number of frames
function async_frame_set(d, frames = 7)
	v = vec(first(first(d)))
	lv = length(v)
	fl = Int(lv / frames)
	BEGIN = Int((frames - 1) / 2) * fl + 1
	END = (BEGIN - 1) + fl
	return [(vec(x), vec(x)[BEGIN:END]) for (x, y) in d]
end

# ╔═╡ 783150a9-2578-4eee-b277-0a317464a299
begin
	fname_dev = "development_chunks_refined.csv"
	fname_eval = "evaluation_chunks_refined.csv"
	
	dev_chunks = get_chunk_set(fname_dev)
	eval_chunks = get_chunk_set(fname_eval)
end

# ╔═╡ e3135a53-14e0-46f6-8a5a-3f0450dc4014
begin
	RAW_FTR = "data"
	MFCC_FTR = "mfcc"
	MBK_FTR = "mbk"
	
	# trn_raw = chunk_set_loader_unbatched(dev_chunks; feature=RAW_FTR) 
	# tst_raw = chunk_set_loader_unbatched(eval_chunks; feature=RAW_FTR)
	# trn_mbk = chunk_set_loader_unbatched(dev_chunks; feature=MBK_FTR) 
	# tst_mbk = chunk_set_loader_unbatched(eval_chunks; feature=MBK_FTR)
	# trn_mfcc = chunk_set_loader_unbatched(dev_chunks; feature=MFCC_FTR) 
	# tst_mfcc = chunk_set_loader_unbatched(eval_chunks; feature=MFCC_FTR)
	trn_mfcc = chunk_set_loader_batched(dev_chunks; feature=MFCC_FTR) 
	tst_mfcc = chunk_set_loader_batched(eval_chunks; feature=MFCC_FTR)
end

# ╔═╡ 20f2b436-d56a-4a99-96d7-0c55973e20be
# begin
# 	raw_dnn = DNN(64000) |> device
# 	raw_dnn_p = params(raw_dnn)
# 	bce_dnn_raw_loss = bdl(Flux.Losses.binarycrossentropy, raw_dnn)

# 	# smoke-test training DNN
# 	for i in 1:nepochs
# 		println("RAW DNN epoch $i")
# 		Flux.train!(bce_dnn_raw_loss, raw_dnn_p, trn_raw, opt(); 
# 			cb=cb(tst_loss_batched(tst_raw, bce_dnn_raw_loss)))
# 	end
# end

# ╔═╡ 07beeaa0-7514-4138-9cec-909868edfe7d
begin
	mfcc_dnn = DNN(399*24) |> device
	mfcc_dnn_p = params(mfcc_dnn)
	bce_dnn_mfcc_loss = bdl(Flux.Losses.binarycrossentropy, mfcc_dnn)
	
	for i in 1:nepochs
		println("MFCC DNN epoch $i")
		Flux.train!(bce_dnn_mfcc_loss, mfcc_dnn_p, trn_mfcc, opt(); 
			cb=cb(tst_loss_batched(tst_mfcc, bce_dnn_mfcc_loss)))
	end
end

# ╔═╡ d698686b-ee9e-4a95-91f4-f03e9c59df26
# begin
# 	mbk_dnn = DNN(399 * 40) |> device
# 	mbk_dnn_p = params(mbk_dnn)
# 	bce_dnn_mbk_loss = bdl(Flux.Losses.binarycrossentropy, mbk_dnn)

# 	for i in 1:nepochs
# 		println("MBK DNN epoch $i")
# 		Flux.train!(bce_dnn_mbk_loss, mbk_dnn_p, trn_mbk, opt(); 
# 			cb=cb(tst_loss_batched(tst_mbk, bce_dnn_mbk_loss)))
# 	end
# end

# ╔═╡ 45e998a5-d75f-4349-91f8-5fc1550913ce
# begin
# 	adae = aDAE(399 * 40) |> device
# 	adae_p = params(adae)

# 	adae_loss = udl(Flux.Losses.mse, adae)
# 	adae_tst_loss = () -> sum([adae_loss(d...) for d in async_frame_set(tst_mbk)])
	
# 	for i in 1:nepochs
# 		println("aDAE epoch $i")
# 		Flux.train!(
# 			adae_loss, adae_p, async_frame_set(trn_mbk),
# 			opt(), cb=cb(adae_tst_loss)
# 		)
# 	end
# end

# ╔═╡ 5a46e054-de91-4fad-933d-e0459bce40e0
# begin
# 	adae_dnn = DNN(50) |> device
# 	adae_dnn_p = params(adae_dnn)

# 	function adae_dnn_loss(x, y)
# 		yhat = x |> device |> adae[1:4] |> adae_dnn
# 		return Flux.Losses.binarycrossentropy(yhat, y |> device)
# 	end

# 	for i in 1:nepochs
# 		println("aDAE_DNN epoch $i")
# 		Flux.train!(
# 			adae_dnn_loss, adae_dnn_p, trn_mbk, opt(),
# 			cb=cb(tst_loss(tst_mbk, adae_dnn_loss))
# 		)
# 	end
# end

# ╔═╡ Cell order:
# ╠═b0bd58fb-3758-4909-a337-020ffb1752e4
# ╠═bd650de6-7ee1-454f-8d53-270b43f70770
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
# ╠═6a5c11a3-a8d4-4491-9637-8472cc076861
# ╠═9591981b-63ea-4edf-9cef-2760963ff3ed
# ╠═c9e50684-8e04-4481-a2fe-8612c1c6e8fd
# ╠═64856590-dd0e-4708-96b2-e776bd46796a
# ╠═783150a9-2578-4eee-b277-0a317464a299
# ╠═e3135a53-14e0-46f6-8a5a-3f0450dc4014
# ╠═20f2b436-d56a-4a99-96d7-0c55973e20be
# ╠═07beeaa0-7514-4138-9cec-909868edfe7d
# ╠═d698686b-ee9e-4a95-91f4-f03e9c59df26
# ╠═45e998a5-d75f-4349-91f8-5fc1550913ce
# ╠═5a46e054-de91-4fad-933d-e0459bce40e0
