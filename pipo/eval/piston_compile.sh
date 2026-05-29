#!/bin/bash

# reference: https://codeforces.com/blog/entry/121114

set -e
case "${PISTON_LANGUAGE}" in
    c++)
        # GNU C++
        g++ -DONLINE_JUDGE -O2 -pipe -s $1
        ;;
    c++0x)
        # GNU C++0x
        g++ -DONLINE_JUDGE -O2 -std=c++0x -pipe -s $1
        ;;
    c++11)
        # GNU C++11
        g++ -DONLINE_JUDGE -O2 -std=c++11 -pipe -s $1
        ;;
    c++14)
        # C++14 (GCC 6-32)
        g++ -DONLINE_JUDGE -O2 -std=c++14 -pipe -s $1
        ;;
    c++17)
        # C++17 (GCC 7-32 and GCC 9-64)
        g++ -DONLINE_JUDGE -O2 -std=c++17 -pipe -s $1
        ;;
    c++20)
        # C++20 (GCC 11-64 and GCC 13-64)
        g++ -Wall -Wextra -Wconversion -DONLINE_JUDGE -O2 -std=c++20 -pipe -s $1
        ;;
    python3)
        # Python 3
        echo "Skipping compile - python3"
        exit 0
        ;;
    kotlin)
        # Kotlin 1.4
        kotlinc $1 -include-runtime -d main.jar
        ;;
    java)
        main_class=$(grep -m 1 -E "^[^/]*public (final|abstract)? *class" main.java | sed -E 's/.*class +([A-Za-z0-9_]+).*/\1/')
        echo "Found main class: $main_class"
        echo "Renaming main.java to $main_class.java"
        mv main.java $main_class.java
        echo "Compiling $main_class.java"
        # Java
        javac -Xlint:unchecked $main_class.java
        ;;
    *)
        echo "Unsupported language: ${PISTON_LANGUAGE}"
        exit 1
        ;;
esac

# For compiled languages that produce an a.out file
if [ -f "a.out" ]; then
    chmod +x a.out
fi
