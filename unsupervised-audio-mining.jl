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
function chunk_set_loader_batched(df; feature, batchsize=100)
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
			loss = 0
			for (x, y) in zip(d.data, d.labels)
				loss += lss(mdl(x |> device), y |> device)
			end
			return loss
			# xs = reduce(hcat, d.data) |> device
			# ys = reduce(hcat, d.labels) |> device
			# yhats = mdl(xs) |> device
			# return sum(lss.(yhats, ys))
		end
	end
	
	opt = () -> Flux.Optimise.Momentum(0.005, 0.9)
	# opt = () -> Flux.Optimise.ADAM()
	
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
	
	trn_raw = chunk_set_loader_unbatched(dev_chunks; feature=RAW_FTR) 
	tst_raw = chunk_set_loader_unbatched(eval_chunks; feature=RAW_FTR)
	trn_mbk = chunk_set_loader_unbatched(dev_chunks; feature=MBK_FTR) 
	tst_mbk = chunk_set_loader_unbatched(eval_chunks; feature=MBK_FTR)
	trn_mfcc = chunk_set_loader_unbatched(dev_chunks; feature=MFCC_FTR) 
	tst_mfcc = chunk_set_loader_unbatched(eval_chunks; feature=MFCC_FTR)
end

# ╔═╡ 20f2b436-d56a-4a99-96d7-0c55973e20be
begin
	raw_dnn = DNN(64000) |> device
	raw_dnn_p = params(raw_dnn)
	bce_dnn_raw_loss = udl(Flux.Losses.binarycrossentropy, raw_dnn)

	for i in 1:nepochs
		println("RAW DNN epoch $i")
		Flux.train!(bce_dnn_raw_loss, raw_dnn_p, trn_raw, opt(); 
			cb=cb(tst_loss_unbatched(tst_raw, bce_dnn_raw_loss)))
	end
end

# ╔═╡ 07beeaa0-7514-4138-9cec-909868edfe7d
begin
	mfcc_dnn = DNN(399*24) |> device
	mfcc_dnn_p = params(mfcc_dnn)
	bce_dnn_mfcc_loss = udl(Flux.Losses.binarycrossentropy, mfcc_dnn)
	
	for i in 1:nepochs
		println("MFCC DNN epoch $i")
		Flux.train!(bce_dnn_mfcc_loss, mfcc_dnn_p, trn_mfcc, opt(); 
			cb=cb(tst_loss_unbatched(tst_mfcc, bce_dnn_mfcc_loss)))
	end
end

# ╔═╡ d698686b-ee9e-4a95-91f4-f03e9c59df26
begin
	mbk_dnn = DNN(399 * 40) |> device
	mbk_dnn_p = params(mbk_dnn)
	bce_dnn_mbk_loss = udl(Flux.Losses.binarycrossentropy, mbk_dnn)

	for i in 1:nepochs
		println("MBK DNN epoch $i")
		Flux.train!(bce_dnn_mbk_loss, mbk_dnn_p, trn_mbk, opt(); 
			cb=cb(tst_loss_unbatched(tst_mbk, bce_dnn_mbk_loss)))
	end
end

# ╔═╡ 45e998a5-d75f-4349-91f8-5fc1550913ce
begin
	adae = aDAE(399 * 40) |> device
	adae_p = params(adae)

	adae_loss = udl(Flux.Losses.mse, adae)
	adae_tst_loss = () -> sum([adae_loss(d...) for d in async_frame_set(tst_mbk)])
	
	for i in 1:nepochs
		println("aDAE epoch $i")
		Flux.train!(
			adae_loss, adae_p, async_frame_set(trn_mbk),
			opt(), cb=cb(adae_tst_loss)
		)
	end
end

# ╔═╡ 5a46e054-de91-4fad-933d-e0459bce40e0
begin
	adae_dnn = DNN(50) |> device
	adae_dnn_p = params(adae_dnn)

	function adae_dnn_loss(x, y)
		yhat = x |> device |> adae[1:4] |> adae_dnn
		return Flux.Losses.binarycrossentropy(yhat, y |> device)
	end

	for i in 1:nepochs
		println("aDAE_DNN epoch $i")
		Flux.train!(
			adae_dnn_loss, adae_dnn_p, trn_mbk, opt(),
			cb=cb(tst_loss(tst_mbk, adae_dnn_loss))
		)
	end
end

# ╔═╡ Cell order:
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
