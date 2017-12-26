#!/bin/bash
cd /code/python
source activate spike-ring
python spike_ring/server.py -D
sleep 60
cd /code
java -jar spike-ring.jar -D
