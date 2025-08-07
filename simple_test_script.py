#!/usr/bin/env python3
"""
Simple test script to debug generation issues
Run this BEFORE your main training to identify problems
"""

import torch
import torch.nn as nn
from smiles_tokenizer import SMILESTokenizer, create_tokenizer

# Create a minimal test case
def test_tokenizer_and_generation():
    print("="*60)
    print("DEBUGGING TOKENIZER AND GENERATION")
    print("="*60)
    
    # 1. Test tokenizer creation
    print("\n1. Testing tokenizer creation...")
    test_smiles = ["CCO", "CC(C)C", "c1ccccc1", "CCN(C)C"]
    
    try:
        tokenizer = create_tokenizer(test_smiles, min_freq=1)
        print(f"✓ Tokenizer created successfully")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        
        # Debug tokenizer
        if hasattr(tokenizer, 'token_to_id'):
            print(f"  Special tokens available:")
            special = ['<pad>', '<bos>', '<eos>', '<unk>']
            for token in special:
                if token in tokenizer.token_to_id:
                    print(f"    {token}: {tokenizer.token_to_id[token]}")
                else:
                    print(f"    {token}: NOT FOUND")
        
        # Test encode/decode
        print(f"\n2. Testing encode/decode...")
        for smiles in test_smiles[:2]:
            print(f"  Testing: {smiles}")
            try:
                encoded = tokenizer.encode(smiles, max_length=50)
                decoded = tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)
                print(f"    Encoded: {encoded['input_ids'][:10]}...")
                print(f"    Decoded: '{decoded}'")
                print(f"    Match: {smiles == decoded}")
            except Exception as e:
                print(f"    ERROR: {e}")
        
    except Exception as e:
        print(f"✗ Tokenizer creation failed: {e}")
        return None
    
    # 2. Test minimal model
    print(f"\n3. Testing minimal generation model...")
    
    class MinimalGenerationTest(nn.Module):
        def __init__(self, vocab_size, hidden_dim=256):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.vocab_size = vocab_size
            
            self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
            self.position_embedding = nn.Embedding(100, hidden_dim)  # Small for test
            self.output_projection = nn.Linear(hidden_dim, vocab_size)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, input_ids):
            seq_len = input_ids.size(1)
            batch_size = input_ids.size(0)
            
            positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            
            token_emb = self.token_embedding(input_ids)
            pos_emb = self.position_embedding(positions)
            hidden = self.dropout(token_emb + pos_emb)
            
            output = self.output_projection(hidden)
            return output
        
        def generate_simple(self, tokenizer, max_length=20):
            """Simple generation test"""
            device = next(self.parameters()).device
            
            # Get BOS token
            if hasattr(tokenizer, 'token_to_id') and '<bos>' in tokenizer.token_to_id:
                bos_id = tokenizer.token_to_id['<bos>']
            else:
                bos_id = 1  # Fallback
            
            # Get EOS token  
            if hasattr(tokenizer, 'token_to_id') and '<eos>' in tokenizer.token_to_id:
                eos_id = tokenizer.token_to_id['<eos>']
            else:
                eos_id = 2  # Fallback
                
            print(f"    Using BOS={bos_id}, EOS={eos_id}")
            
            generated = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            
            for step in range(max_length - 1):
                # Forward pass
                logits = self.forward(generated)
                next_logits = logits[0, -1, :]  # Last token
                
                # Sample next token (greedy for now)
                next_token = torch.argmax(next_logits).unsqueeze(0).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop at EOS
                if next_token.item() == eos_id:
                    break
                
                print(f"      Step {step}: token {next_token.item()}")
            
            return generated
    
    try:
        # Create test model
        test_model = MinimalGenerationTest(tokenizer.vocab_size)
        print(f"  ✓ Model created with vocab_size={tokenizer.vocab_size}")
        
        # Test generation
        print(f"  Testing generation...")
        with torch.no_grad():
            generated = test_model.generate_simple(tokenizer)
            token_ids = generated[0].tolist()
            
            print(f"    Generated token IDs: {token_ids}")
            
            # Try to decode
            try:
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                print(f"    Decoded: '{decoded}'")
                
                if decoded.strip() == "":
                    print(f"    ⚠ WARNING: Empty decoded string!")
                    print(f"    Checking individual tokens:")
                    if hasattr(tokenizer, 'id_to_token'):
                        for i, token_id in enumerate(token_ids):
                            if token_id in tokenizer.id_to_token:
                                token = tokenizer.id_to_token[token_id]
                                print(f"      {i}: {token_id} -> '{token}'")
                            else:
                                print(f"      {i}: {token_id} -> UNKNOWN")
                else:
                    print(f"    ✓ Generation successful!")
                    
            except Exception as e:
                print(f"    ✗ Decoding failed: {e}")
        
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Debug complete. Check the output above for issues.")
    print("="*60)
    
    return tokenizer

if __name__ == "__main__":
    # Run the test
    tokenizer = test_tokenizer_and_generation()
    
    if tokenizer is None:
        print("CRITICAL: Tokenizer test failed. Fix tokenizer issues first!")
    else:
        print(f"Tokenizer test passed. Vocab size: {tokenizer.vocab_size}")
        print("You can now proceed with training, but check for any warnings above.")