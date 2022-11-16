#!/bin/bash

make clean && make


echo "Size: 9000x35000 * 35000x5750"
./sparsematmult 9000 35000 5750 0.05 | tee 9000x35000*35000x5750_0.05.log
./sparsematmult 9000 35000 5750 0.10 | tee 9000x35000*35000x5750_0.10.log
./sparsematmult 9000 35000 5750 0.15 | tee 9000x35000*35000x5750_0.15.log
./sparsematmult 9000 35000 5750 0.20 | tee 9000x35000*35000x5750_0.20.log
echo ""

