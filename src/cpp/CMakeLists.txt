cmake_minimum_required(VERSION 3.23)
project(BattleSnakeCpp)

set(CMAKE_CXX_STANDARD 17)
set(SOURCE_FILES main.cpp)

add_executable(BattleSnakeCpp main.cpp
        source/battlesnake.cpp
        source/battlesnake_helper.cpp
        source/nash.cpp
        source/logit.cpp
        source/utils.cpp
        header/battlesnake.h
        header/battlesnake_helper.h
        header/nash.h
        header/logit.h
        header/utils.h
        link.cpp
        alglib/alglibinternal.cpp
        alglib/alglibinternal.h
        alglib/alglibmisc.cpp
        alglib/alglibmisc.h
        alglib/ap.cpp
        alglib/ap.h
        alglib/linalg.cpp
        alglib/linalg.h
        alglib/optimization.cpp
        alglib/optimization.h
        alglib/solvers.cpp
        alglib/solvers.h
        header/mle.h source/mle.cpp header/quantal.h source/quantal.cpp
        source/logit.cpp header/overcooked.h source/overcooked.cpp)
