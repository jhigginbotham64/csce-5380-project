# runs miniconda jupyter notebook on default port.
# usually this is 8888, but if you're connecting
# remotely then edit this script to make sure it
# runs on the same port as your SSH tunnel. also,
# if you're not able to connect then make sure you
# get the right URL token from stdout.
~/.julia/conda/3/bin/jupyter notebook --no-browser &
