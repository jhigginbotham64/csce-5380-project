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
	using MFCC
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

# ╔═╡ 52518e6f-d42b-45bf-a5be-b02e899c88e5
function preprocess_chunk(chunkname)
	c = CSV.read(
	   IOBuffer(open(String, chime_home["chunks"][chunkname * ".csv"])), 
	   DataFrame; header=["key","val"])
	c = Dict(key => getdfval(c, key) for key in c.key)
	c["c"] = 'c' in c["majorityvote"] ? 1 : 0
	c["m"] = 'm' in c["majorityvote"] ? 1 : 0
	c["f"] = 'f' in c["majorityvote"] ? 1 : 0
	c["v"] = 'v' in c["majorityvote"] ? 1 : 0
	c["p"] = 'p' in c["majorityvote"] ? 1 : 0
	c["b"] = 'b' in c["majorityvote"] ? 1 : 0
	c["o"] = 'o' in c["majorityvote"] ? 1 : 0
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
function getchunkdf(df)
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

# ╔═╡ b21ff8cc-d39b-48dd-b6af-ced0cca5608a
begin
	chunk_headers=["id", "chunkname"];
	# eval_chunks = getchunkdf(CSV.read(
	# 	IOBuffer(open(String, chime_home["evaluation_chunks_refined.csv"])), 
	# 	DataFrame; header=chunk_headers));
	dev_chunks = getchunkdf(CSV.read(
		IOBuffer(open(String, chime_home["development_chunks_refined.csv"])), 
		DataFrame; header=chunk_headers));
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
# ╠═b21ff8cc-d39b-48dd-b6af-ced0cca5608a
# ╠═e8d57d33-a224-4150-9308-69455db18465
