# install julia
wget -c https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz -O - | tar xz

# get packages and set up IJulia
echo -e "\n" | ./julia-1.6.1/bin/julia --project=@. -e "using Pkg; Pkg.update(); using IJulia; notebook(detached=true)"

# kill notebook, not reachable by default so we need to start properly later anyway
killall -9 jupyter-notebook
# if the killall command is not available, you may need to install it:
# sudo apt install psmisc

# install chime_home dataset
wget -c https://archive.org/download/chime-home/chime_home.tar.gz -O - | tar xz
