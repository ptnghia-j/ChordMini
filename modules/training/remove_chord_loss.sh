#!/bin/bash

# This script safely removes the ChordLoss.py file from the codebase
# Run with: bash remove_chord_loss.sh

CHORD_LOSS_PATH="/Users/nghiaphan/Desktop/ChordMini/modules/training/ChordLoss.py"

# Check if file exists
if [ -f "$CHORD_LOSS_PATH" ]; then
    # Create backup
    echo "Creating backup of ChordLoss.py"
    cp "$CHORD_LOSS_PATH" "${CHORD_LOSS_PATH}.bak"
    
    # Remove the file
    echo "Removing ChordLoss.py"
    rm "$CHORD_LOSS_PATH"
    
    echo "ChordLoss.py has been removed. A backup was created at ${CHORD_LOSS_PATH}.bak"
else
    echo "ChordLoss.py not found at $CHORD_LOSS_PATH"
fi

# Remove chord loss from all imports in the project
echo "Removing all imports of ChordLoss from Python files"
find "/Users/nghiaphan/Desktop/ChordMini" -name "*.py" -type f -exec grep -l "from modules.training.ChordLoss" {} \; | xargs sed -i '' '/from modules.training.ChordLoss/d'

echo "Chord loss removal completed"
