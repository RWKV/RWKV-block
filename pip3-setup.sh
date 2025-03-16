#!/bin/bash

# # Install croc (for file transfer)
# curl https://getcroc.schollz.com | bash

# Setup submodules
git submodule update --init --recursive

# Install dependencies
pip3 install -r requirements.txt

# Including test dependencies
$(cd test && pip3 install -r requirements.txt)
