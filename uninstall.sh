# this script is mostly useful for showing you
# what things get installed by install.sh and
# how you can remove them. don't run it directly
# unless you know what you're doing.

# local julia install
rm -rf julia-1.6.1/

# julia home (do not remove if you have an existing julia installation)
rm -rf ~/.julia

# jupyter (ditto if you otherwise use jupyter)
rm -rf ~/.jupyter

# data
rm -rf chime_home/
