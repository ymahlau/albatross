::Compilation of Alglib library
:: Run in Msys using cmd < windows_compile.bat
ECHO Started Compilation of Alglib
g++ -w -std=c++11 -c -fPIC -O3 ap.cpp -o ../compiled/ap.o
g++ -w -std=c++11 -c -fPIC -O3 alglibinternal.cpp -o ../compiled/alglibinternal.o
g++ -w -std=c++11 -c -fPIC -O3 alglibmisc.cpp -o ../compiled/alglibmisc.o
g++ -w -std=c++11 -c -fPIC -O3 linalg.cpp -o ../compiled/linalg.o
g++ -w -std=c++11 -c -fPIC -O3 solvers.cpp -o ../compiled/solvers.o
g++ -w -std=c++11 -c -fPIC -O3 optimization.cpp -o ../compiled/optimization.o
