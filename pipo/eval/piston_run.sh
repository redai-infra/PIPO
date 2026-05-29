#!/usr/bin/env bash
# disable stack limit so you don't get RE with recursion
ulimit -s unlimited
# some problems have 10MB+ input/output files in their test cases and you might get RE. uncomment if needed
# ulimit -f 2097152

INPUT_MODE="stdio"
if [ -f "grader_config" ]; then
    source grader_config
fi

# Initialize JVM memory options string
JVM_MEMORY_OPTS=""
if [ -n "$MEMORY_LIMIT" ]; then
    # Ensure MEMORY_LIMIT is an integer for JVM flags
    MEMORY_LIMIT_MB=$(printf "%.0f" "$MEMORY_LIMIT")
    JVM_MEMORY_OPTS="-Xmx${MEMORY_LIMIT_MB}M"

    # Set memory limit using ulimit only for non-JVM languages
    if [ "${PISTON_LANGUAGE}" != "java" ] && [ "${PISTON_LANGUAGE}" != "kotlin" ]; then
        # handle floats using awk
        MEMORY_LIMIT_KB=$(awk "BEGIN {printf \"%.0f\", $MEMORY_LIMIT * 1024}")
        # Set the memory limit for the entire script and all child processes
        ulimit -v $MEMORY_LIMIT_KB  # Virtual memory
        ulimit -m $MEMORY_LIMIT_KB  # Resident set size
        ulimit -d $MEMORY_LIMIT_KB  # Data segment size
    fi
fi

case "${PISTON_LANGUAGE}" in
    python3)
        TASK_EXECUTABLE="python3 $1"
        ;;
    kotlin)
        TASK_EXECUTABLE="java -XX:NewRatio=5 -Xms8M -Xss64M $JVM_MEMORY_OPTS -DONLINE_JUDGE=true -Duser.language=en -Duser.region=US -Duser.variant=US -jar main.jar"
        ;;
    java)
        main_class=$(ls *.java 2>/dev/null | head -n 1 | sed 's/\.java$//')
        TASK_EXECUTABLE="java -XX:NewRatio=5 -Xms8M -Xss64M $JVM_MEMORY_OPTS -DONLINE_JUDGE=true -Duser.language=en -Duser.region=US -Duser.variant=US $main_class"
        ;;
    *)
        TASK_EXECUTABLE="./a.out"
        ;;
esac

# Choose a very unlikely string
SENTINEL="__NO_CORRECT_OUTPUT__"

# "Securely" handle the correct output file
CORRECT_OUTPUT="$SENTINEL"
if [ -f "correct_output.txt" ]; then
    # Read the content and immediately remove the file
    CORRECT_OUTPUT=$(cat correct_output.txt)
    rm -f correct_output.txt
fi


# some (old) problems ask the contestants to read from input.txt and write to output.txt directly
if [ "$INPUT_MODE" == "file" ]; then
    SOLUTION_OUTPUT="output.txt"
    SOLUTION_INPUT="input.txt"
else
    # Create a temporary file for solution output
    SOLUTION_OUTPUT=$(mktemp)
    SOLUTION_INPUT=$(mktemp)

    # some solutions will read input.txt for local testing... so we need to rename it
    mv input.txt "$SOLUTION_INPUT"
fi

# Global variables for process tracking
declare -a ALL_PIDS

# Define cleanup function - simplified assuming timeout exists
function cleanup {
    # Kill all tracked processes silently
    exec 2>/dev/null
    for pid in "${ALL_PIDS[@]:-}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    
    # Clean up temporary files
    rm -f "$SOLUTION_OUTPUT" || true
    rm -f "$SOLUTION_INPUT" || true
    exec 2>&2
}

# Set up signal handling
trap cleanup EXIT INT TERM

# Function to handle exit codes consistently across task types
function handle_exit_code {
    local exit_code=$1
    
    # Check for known timeout exit codes:
    # - 124: standard timeout exit code
    # - 137: SIGKILL (128+9), used for hard timeouts
    # - 143: SIGTERM (128+15), can also be used for timeouts
    if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ] || [ $exit_code -eq 143 ]; then
        echo "0"
        echo "Time limit exceeded (${TIME_LIMIT}s)" >&2
        return 124
    elif [ $exit_code -eq 134 ]; then
        echo "0"
        echo "Memory limit exceeded (${MEMORY_LIMIT}MB)" >&2
        return 134
    # All other non-zero exit codes should be treated as runtime errors
    elif [ $exit_code -ne 0 ]; then
        echo "0"
        echo "Runtime error with exit code $exit_code" >&2
        return $exit_code
    fi
    
    # Success case - return 0
    return 0
}

# Function to run a command with timeout (simplified assuming timeout exists)
function run_with_timeout {
    local soft_limit=$1; shift
    
    timeout --preserve-status "$soft_limit" "$@"
    return $?
}

# Function to normalize checker output
# 1. Uppercase whole words "yes"/"no" (case-insensitive)
# 2. Normalize all whitespace sequences to single spaces on one line
function normalize_output {
    local filename="$1"
    # 1. Use sed line-by-line for case-insensitive yes/no replacement (GNU extensions: -E for extended regex, \b for word boundary, i for case-insensitive)
    # 2. Use tr to replace newlines with spaces 
    # 3. Use tr again to squeeze all resulting whitespace sequences into single spaces
    # 4. Use sed to trim leading/trailing space from the final single line
    <"$filename" sed -E 's/\b(yes)\b/YES/ig; s/\b(no)\b/NO/ig' | tr '\n' ' ' | tr -s '[:space:]' ' ' | sed 's/^ //;s/ $//'
}

# Normalize input file to Unix line endings (remove \r)
sed -i 's/\r$//' "$SOLUTION_INPUT"


# Simple batch execution with timeout
# If TIME_LIMIT is not set, just run the command directly
if [ -z "$TIME_LIMIT" ]; then
    if [ "$INPUT_MODE" == "file" ]; then
        $TASK_EXECUTABLE > /dev/null 2>&1
    else
        # stdio, pipe
        $TASK_EXECUTABLE < "$SOLUTION_INPUT" > "$SOLUTION_OUTPUT"
    fi
else
    if [ "$INPUT_MODE" == "file" ]; then
        run_with_timeout "$TIME_LIMIT" $TASK_EXECUTABLE > /dev/null 2>&1
    else
        # stdio, pipe
        run_with_timeout "$TIME_LIMIT" $TASK_EXECUTABLE < "$SOLUTION_INPUT" > "$SOLUTION_OUTPUT"
    fi
fi
exit_code=$?

# Handle non-zero exit codes
handle_exit_code $exit_code
rc=$?
if [ $rc -ne 0 ]; then
    exit $rc
fi

# remove empty newlines
sed -i '/^$/d' "$SOLUTION_OUTPUT"
# Check if the variable is still the sentinel value
if [ "$CORRECT_OUTPUT" != "$SENTINEL" ]; then
    # The file existed (and CORRECT_OUTPUT holds its content, possibly "")
    echo "$CORRECT_OUTPUT" > correct_output.txt
    sed -i 's/\r$//' correct_output.txt  # dumb \r from codeforces

    # Remove all empty lines (lines containing only whitespace don't count as empty here).
    # The pattern /^$/ matches lines that start (^) and immediately end ($), meaning they are empty.
    # The 'd' command deletes these matched lines. The -i flag modifies the file in-place.
    # This effectively removes blank lines resulting from consecutive newlines (\n\n becomes \n after removing the empty line between them).
    sed -i '/^$/d' correct_output.txt
    
    # Check if there's a custom checker
    if [ -f "checker.py" ]; then
        # Let the checker handle everything and capture stdout
        result=$(run_with_timeout 10 python3 checker.py "$SOLUTION_INPUT" correct_output.txt "$SOLUTION_OUTPUT")
        exit_code=$?
        # Trim whitespace and check result
        result=$(echo "$result" | tr -d '[:space:]')
        if [ "$result" = "100" ] || [ "$result" = "1" ]; then
            echo "1"
            echo "Output is correct (checker)" >&2
        else
            echo "0"
            echo "Output isn't correct (checker)" >&2
        fi
        exit $exit_code
    else
        # Simple diff-based checking
        # Use cat | xargs to normalize all whitespace to single spaces on one line.
        # diff -b handles potential leading/trailing space differences.
        # add -q to suppress output
        if diff -bq <(normalize_output correct_output.txt) <(normalize_output "$SOLUTION_OUTPUT") >&2; then
            echo "1"
            echo "Output is correct (diff)" >&2
        else
            echo "0"
            echo "Output isn't correct (diff)" >&2
        fi
    fi
else
    # If no correct output was provided, just output the solution's output
    cat "$SOLUTION_OUTPUT"
fi

