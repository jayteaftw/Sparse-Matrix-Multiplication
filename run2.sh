#!/bin/bash

make clean && make


echo "Size: 20000x5300 * 5300x50000"
./sparsematmult 20000 5300 50000 0.05 | tee 20000x5300*5300x50000_0.05.log
./sparsematmult 20000 5300 50000 0.10 | tee 20000x5300*5300x50000_0.10.log
./sparsematmult 20000 5300 50000 0.15 | tee 20000x5300*5300x50000_0.15.log
./sparsematmult 20000 5300 50000 0.20 | tee 20000x5300*5300x50000_0.20.log
echo ""

