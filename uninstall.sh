# uninstall.sh

# this script is mostly useful for showing you
# what things get installed by install.sh and
# how you can remove them. don't run it directly
# unless it won't mess up an existing julia or
# jupyter installation.

# local julia install
rm -rf julia-1.6.1/

# julia home
rm -rf ~/.julia

# jupyter home
rm -rf ~/.jupyter

# data
rm -rf chime_home/
