
#!/bin/bash
# Run OOMMF domain-wall simulations for J = 5e10 ... 45e10 A/m².
# Usage: ./OOMMF_Sequential_Run.sh [path/to/oommf.tcl]
# Optional env vars: OOMMF_TCL_PATH, OOMMF_THREADS.

set -e

OOMMF_PATH="${1:-${OOMMF_TCL_PATH:-oommf.tcl}}"
OOMMF_THREADS="${OOMMF_THREADS:-4}"

if ! command -v tclsh &> /dev/null; then
    echo "Error: tclsh could not be found. Please install Tcl."
    exit 1
fi

if [ ! -f "$OOMMF_PATH" ]; then
    echo "Error: OOMMF not found at: $OOMMF_PATH"
    echo ""
    echo "Usage: $0 [path_to_oommf.tcl]"
    echo "   or set environment variable: export OOMMF_TCL_PATH=/path/to/oommf/oommf.tcl"
    exit 1
fi

OOMMF_PATH_ABS="$(cd "$(dirname "$OOMMF_PATH")" && pwd)/$(basename "$OOMMF_PATH")"

echo "================================================"
echo "OOMMF Batch Simulation Runner"
echo "================================================"
echo "OOMMF Path: $OOMMF_PATH_ABS"
echo "Threads: $OOMMF_THREADS"
echo "Working Directory: $(pwd)"
echo "================================================"
echo ""

for n in $(seq 5 5 45); do
    CURRENT_VAL="${n}e10"
    FOLDER="Motion/J_${CURRENT_VAL}"
    MIF_FILE="DW_motion_J_${CURRENT_VAL}.mif"
    
    echo "--- Processing J = $CURRENT_VAL A/m² ---"
    
    if [ ! -d "$FOLDER" ]; then
        echo "Warning: Directory $FOLDER does not exist. Skipping."
        continue
    fi
    
    if [ ! -f "$FOLDER/$MIF_FILE" ]; then
        echo "Warning: File $FOLDER/$MIF_FILE does not exist. Skipping."
        continue
    fi
    
    cd "$FOLDER"
    echo "Executing: tclsh $OOMMF_PATH_ABS boxsi -threads $OOMMF_THREADS $MIF_FILE"
    tclsh "$OOMMF_PATH_ABS" boxsi -threads "$OOMMF_THREADS" "$MIF_FILE"
    cd - > /dev/null
    
    echo "Completed: $CURRENT_VAL"
    echo ""
done

echo "================================================"
echo "All simulations completed successfully"
echo "================================================"

