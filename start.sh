#!/bin/bash
export EMBED_DEVICE=cpu
cd ~/memory-server
rm -f server.log
nohup .venv/bin/python server.py > server.log 2>&1 &
echo "Started server pid=$!"
