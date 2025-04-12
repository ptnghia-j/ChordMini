import argparse
import torch
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Fix model checkpoint class count mismatch")
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the checkpoint to fix')
    parser.add_argument('--output', type=str, required=True,
                      help='Path for the fixed checkpoint')
    parser.add_argument('--target_classes', type=int, default=170,
                      help='Target number of classes (default: 170)')
    args = parser.parse_args()
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Extract the state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        # Reconstruct checkpoint as state_dict style
        checkpoint = {'model_state_dict': state_dict}
    
    # Check output layer dimensions
    if 'fc.weight' in state_dict:
        current_classes = state_dict['fc.weight'].size(0)
        feature_dim = state_dict['fc.weight'].size(1)
        
        print(f"Current model has {current_classes} output classes with feature dimension {feature_dim}")
        print(f"Target number of classes: {args.target_classes}")
        
        if current_classes != args.target_classes:
            print(f"Modifying output layer dimensions from {current_classes} to {args.target_classes}...")
            
            # Create new tensors with correct dimensions
            if args.target_classes > current_classes:
                # Expanding - create new tensors and copy old values
                new_weight = torch.zeros((args.target_classes, feature_dim), 
                                        dtype=state_dict['fc.weight'].dtype)
                new_bias = torch.zeros(args.target_classes, 
                                      dtype=state_dict['fc.bias'].dtype)
                
                # Copy existing values
                new_weight[:current_classes, :] = state_dict['fc.weight']
                new_bias[:current_classes] = state_dict['fc.bias']
                
                # Initialize the remaining weights with small random values
                if current_classes < args.target_classes:
                    torch.nn.init.xavier_uniform_(new_weight[current_classes:, :])
                    new_bias[current_classes:].fill_(0.0)
            else:
                # Truncating - just keep the first n classes
                new_weight = state_dict['fc.weight'][:args.target_classes, :]
                new_bias = state_dict['fc.bias'][:args.target_classes]
            
            # Update the state dict
            state_dict['fc.weight'] = new_weight
            state_dict['fc.bias'] = new_bias
            
            # Save the checkpoint
            if 'model_state_dict' in checkpoint:
                checkpoint['model_state_dict'] = state_dict
            elif 'model' in checkpoint:
                checkpoint['model'] = state_dict
            else:
                checkpoint = state_dict
            
            # Add metadata about the modification
            if isinstance(checkpoint, dict) and checkpoint != state_dict:
                checkpoint['n_classes'] = args.target_classes
                checkpoint['modified'] = True
                
            print(f"Saving modified checkpoint to {args.output}")
            torch.save(checkpoint, args.output)
            print("Done!")
        else:
            print("No modifications needed, output classes already match target.")
    else:
        print("ERROR: Could not find 'fc.weight' in the state dictionary. Is this a valid ChordNet checkpoint?")
        sys.exit(1)

if __name__ == "__main__":
    main()
