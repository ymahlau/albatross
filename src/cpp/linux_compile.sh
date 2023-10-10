#!/bin/bash
# Compilation of C++ Environment
# Run with sh linux_compile.sh
echo "Started Compilation"
g++ -Wall -std=c++11 -c -fPIC -O3 source/battlesnake.cpp -o compiled/battlesnake.o 
g++ -Wall -std=c++11 -c -fPIC -O3 source/battlesnake_helper.cpp -o compiled/battlesnake_helper.o 
g++ -Wall -std=c++11 -c -fPIC -O3 source/utils.cpp -o compiled/utils.o 
g++ -Wall -std=c++11 -c -fPIC -O3 source/nash.cpp -o compiled/nash.o 
g++ -Wall -std=c++11 -c -fPIC -O3 source/logit.cpp -o compiled/logit.o
g++ -Wall -std=c++11 -c -fPIC -O3 source/mle.cpp -o compiled/mle.o 
g++ -Wall -std=c++11 -c -fPIC -O3 source/quantal.cpp -o compiled/quantal.o 
g++ -Wall -std=c++11 -c -fPIC -O3 link.cpp -o compiled/link.o 
echo "Starting Linking..."
sudo g++ -Wall -std=c++11 -O3 -shared -Wl,-soname -o compiled/liblink.so \
 compiled/ap.o \
 compiled/alglibinternal.o \
 compiled/alglibmisc.o \
 compiled/linalg.o \
 compiled/solvers.o \
 compiled/optimization.o \
 compiled/utils.o \
 compiled/logit.o \
 compiled/nash.o \
 compiled/battlesnake.o \
 compiled/battlesnake_helper.o \
 compiled/mle.o \
 compiled/quantal.o \
 compiled/link.o
