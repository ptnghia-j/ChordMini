#!/bin/bash
echo "Cleaning Git history of large files..."
echo "This will PERMANENTLY REWRITE your Git history."
echo "Make sure you've pushed all your changes to a backup branch first!"
read -p "Are you sure you want to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation canceled."
    exit 1
fi

# Create a backup branch
CURRENT_BRANCH=$(git branch --show-current)
git branch backup-before-cleaning-$(date +%Y%m%d)

echo "Created backup branch: backup-before-cleaning-$(date +%Y%m%d)"
echo "Starting cleanup process. This might take a while..."

# Step 1: Remove large model files
echo "Removing PyTorch model files..."
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch '*.pt' '*.pth' '*.ckpt' '*.bin'" \
  --prune-empty --tag-name-filter cat -- --all

# Step 2: Remove audio files
echo "Removing audio files..."
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch '*.mp3' '*.wav' '*.ogg' '*.flac' '*.m4a'" \
  --prune-empty --tag-name-filter cat -- --all

# Step 3: Remove NumPy cache files
echo "Removing NumPy cache files..."
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch '*.npy' 'cache/**/*.npy'" \
  --prune-empty --tag-name-filter cat -- --all

# Step 4: Specifically remove the large files identified earlier
echo "Removing specifically identified large files..."
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch 'modules/models/Teacher/BTC/test/btc_model_large_voca.pt' 'modules/models/Teacher/BTC/test/btc_model.pt' 'test/mhwgo.mp3'" \
  --prune-empty --tag-name-filter cat -- --all

# Step 5: Prune and clean
echo "Cleaning and pruning..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Done! Your repository should now be much smaller."
echo "To complete the process, force push to your remote repository:"
echo "git push origin $CURRENT_BRANCH --force"
echo ""
echo "IMPORTANT: All collaborators will need to re-clone the repository!"
