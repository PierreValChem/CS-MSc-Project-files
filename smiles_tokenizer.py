"""
SMILES Tokenizer for NMR-to-SMILES prediction
Save this as: smiles_tokenizer.py
"""

import re
import json
import torch
from typing import List, Dict, Union, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class SMILESTokenizer:
    """
    Optimized SMILES tokenizer that respects chemical grammar
    Better than ChemBERTa for generation tasks
    """
    
    def __init__(self, vocab_file: Optional[str] = None):
        # Comprehensive SMILES regex pattern
        self.pattern = re.compile(
            r'(\[[^\]]+\]|'  # Bracketed atoms (stereochemistry, charges, isotopes)
            r'Br|Cl|Si|Se|se|As|as|'  # Two-letter elements
            r'@@|@|'  # Stereochemistry markers
            r'%\d{2}|%\d{3}|'  # Ring closures (up to 3 digits for complex molecules)
            r'[BCNOPSFIbcnopsfi]|'  # Single letter elements
            r'[0-9]|'  # Single digit ring closures
            r'[+\-]|'  # Charges
            r'[=#$:~/\\\\.]|'  # Bonds (including aromatic :)
            r'[()])'  # Parentheses
        )
        
        # Special tokens
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>', '<mask>']
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        self.mask_token = '<mask>'
        
        # Token to ID mappings
        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            self.token2id = {token: i for i, token in enumerate(self.special_tokens)}
            self.id2token = {i: token for token, i in self.token2id.items()}
        
        # Set special token IDs
        self.pad_token_id = self.token2id[self.pad_token]
        self.sos_token_id = self.token2id[self.sos_token]
        self.eos_token_id = self.token2id[self.eos_token]
        self.unk_token_id = self.token2id[self.unk_token]
        self.mask_token_id = self.token2id[self.mask_token]
    
    @property
    def vocab_size(self) -> int:
        return len(self.token2id)
    
    def tokenize(self, smiles: str, add_special_tokens: bool = False) -> List[str]:
        """Tokenize SMILES string into chemically meaningful tokens"""
        tokens = self.pattern.findall(smiles)
        
        if add_special_tokens:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        
        return tokens
    
    def build_vocab(self, smiles_list: List[str], min_freq: int = 2) -> None:
        """Build vocabulary from a list of SMILES strings"""
        logger.info("Building vocabulary from SMILES dataset...")
        
        # Count token frequencies
        token_counter = Counter()
        for smiles in smiles_list:
            tokens = self.tokenize(smiles, add_special_tokens=False)
            token_counter.update(tokens)
        
        # Initialize vocab with special tokens
        self.token2id = {token: i for i, token in enumerate(self.special_tokens)}
        
        # Add tokens that appear at least min_freq times
        for token, count in token_counter.items():
            if count >= min_freq and token not in self.token2id:
                self.token2id[token] = len(self.token2id)
        
        # Create reverse mapping
        self.id2token = {i: token for token, i in self.token2id.items()}
        
        logger.info(f"Vocabulary size: {len(self.token2id)} tokens")
        logger.info(f"Most common tokens: {token_counter.most_common(20)}")
        
        # Analyze vocabulary coverage
        total_tokens = sum(token_counter.values())
        covered_tokens = sum(count for token, count in token_counter.items() 
                           if token in self.token2id)
        coverage = covered_tokens / total_tokens * 100
        logger.info(f"Vocabulary coverage: {coverage:.2f}%")
    
    def encode(self, smiles: str, max_length: int = 256, 
               padding: bool = True, truncation: bool = True,
               return_tensors: Optional[str] = 'pt') -> Dict[str, Union[List[int], torch.Tensor]]:
        """Encode SMILES string to token IDs"""
        # Tokenize
        tokens = self.tokenize(smiles, add_special_tokens=True)
        
        # Convert to IDs
        input_ids = []
        for token in tokens:
            if token in self.token2id:
                input_ids.append(self.token2id[token])
            else:
                input_ids.append(self.unk_token_id)
        
        # Truncate if necessary
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length-1] + [self.eos_token_id]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad if necessary
        if padding and len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], 
               skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to SMILES string"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = []
        for id in token_ids:
            if id in self.id2token:
                token = self.id2token[id]
                
                if skip_special_tokens and token in self.special_tokens:
                    if token == self.eos_token:
                        break
                    continue
                
                tokens.append(token)
        
        return ''.join(tokens)
    
    def batch_encode(self, smiles_list: List[str], max_length: int = 256, 
                    padding: bool = True, truncation: bool = True,
                    return_tensors: Optional[str] = 'pt') -> Dict[str, torch.Tensor]:
        """Encode a batch of SMILES strings"""
        encoded_batch = []
        
        for smiles in smiles_list:
            encoded = self.encode(
                smiles, 
                max_length=max_length, 
                padding=padding,
                truncation=truncation,
                return_tensors=None  # Get lists first
            )
            encoded_batch.append(encoded)
        
        # Find max length in batch for dynamic padding
        if not padding:
            max_len = max(len(e['input_ids']) for e in encoded_batch)
            # Pad to batch max length
            for encoded in encoded_batch:
                pad_len = max_len - len(encoded['input_ids'])
                encoded['input_ids'].extend([self.pad_token_id] * pad_len)
                encoded['attention_mask'].extend([0] * pad_len)
        
        # Convert to tensors
        if return_tensors == 'pt':
            input_ids = torch.tensor([e['input_ids'] for e in encoded_batch], dtype=torch.long)
            attention_mask = torch.tensor([e['attention_mask'] for e in encoded_batch], dtype=torch.long)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            return {
                'input_ids': [e['input_ids'] for e in encoded_batch],
                'attention_mask': [e['attention_mask'] for e in encoded_batch]
            }
    
    def save_vocab(self, file_path: str) -> None:
        """Save vocabulary to file"""
        vocab_data = {
            'token2id': self.token2id,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }
        with open(file_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        logger.info(f"Vocabulary saved to {file_path}")
    
    def load_vocab(self, file_path: str) -> None:
        """Load vocabulary from file"""
        with open(file_path, 'r') as f:
            vocab_data = json.load(f)
        
        self.token2id = vocab_data['token2id']
        self.special_tokens = vocab_data['special_tokens']
        self.id2token = {int(i): token for token, i in self.token2id.items()}
        
        # Reset special token IDs
        self.pad_token_id = self.token2id[self.pad_token]
        self.sos_token_id = self.token2id[self.sos_token]
        self.eos_token_id = self.token2id[self.eos_token]
        self.unk_token_id = self.token2id[self.unk_token]
        self.mask_token_id = self.token2id[self.mask_token]
        
        logger.info(f"Vocabulary loaded from {file_path} (size: {self.vocab_size})")
    
    # HuggingFace compatibility methods
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert tokens to IDs (HuggingFace compatible)"""
        if isinstance(tokens, str):
            return self.token2id.get(tokens, self.unk_token_id)
        return [self.token2id.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert IDs to tokens (HuggingFace compatible)"""
        if isinstance(ids, int):
            return self.id2token.get(ids, self.unk_token)
        return [self.id2token.get(id, self.unk_token) for id in ids]
    
    def __call__(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Make tokenizer callable (HuggingFace compatible)"""
        if isinstance(text, str):
            return self.encode(text, **kwargs)
        else:
            return self.batch_encode(text, **kwargs)
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return self.vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary"""
        return self.token2id.copy()


def create_tokenizer(train_smiles: List[str], 
                    save_path: str = 'smiles_tokenizer_vocab.json',
                    min_freq: int = 1) -> SMILESTokenizer:
    """
    Convenience function to create and save a tokenizer
    
    Args:
        train_smiles: List of SMILES strings from training data
        save_path: Path to save vocabulary
        min_freq: Minimum frequency for token inclusion
        
    Returns:
        Configured SMILESTokenizer
    """
    tokenizer = SMILESTokenizer()
    tokenizer.build_vocab(train_smiles, min_freq=min_freq)
    tokenizer.save_vocab(save_path)
    
    # Print statistics
    logger.info("\nTokenization Statistics:")
    lengths = []
    for smiles in train_smiles[:1000]:  # Sample first 1000
        tokens = tokenizer.tokenize(smiles, add_special_tokens=True)
        lengths.append(len(tokens))
    
    import numpy as np
    logger.info(f"Average sequence length: {np.mean(lengths):.1f}")
    logger.info(f"Max sequence length: {max(lengths)}")
    logger.info(f"95th percentile length: {np.percentile(lengths, 95):.0f}")
    
    return tokenizer