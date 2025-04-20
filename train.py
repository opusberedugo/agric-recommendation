import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import argparse
import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nlp_training.log"),
        logging.StreamHandler()
    ]
)

# Constants
MAX_SEQ_LENGTH = 100
BATCH_SIZE = 16
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 20

class SimpleTokenizer:
    """A simple tokenizer class similar to Keras Tokenizer"""
    
    def __init__(self, num_words=None, oov_token='<UNK>'):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}
        self.word_counts = {}
        self.document_count = 0
        # Add special tokens
        self.word_index['<PAD>'] = 0  # Padding token
        self.word_index['<UNK>'] = 1  # Unknown token
        self.word_index['<START>'] = 2  # Start of sequence token
        self.word_index['<END>'] = 3  # End of sequence token
        self.index_word = {v: k for k, v in self.word_index.items()}
        
    def fit_on_texts(self, texts):
        """Build vocabulary from list of texts"""
        for text in texts:
            self.document_count += 1
            for word in text.lower().split():
                if word in self.word_counts:
                    self.word_counts[word] += 1
                else:
                    self.word_counts[word] = 1
        
        # Sort words by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Only keep most common words if num_words is specified
        if self.num_words is not None:
            sorted_words = sorted_words[:self.num_words - 4]  # -4 for special tokens
        
        # Create word index
        for word, _ in sorted_words:
            idx = len(self.word_index)
            self.word_index[word] = idx
            self.index_word[idx] = word
            
    def texts_to_sequences(self, texts):
        """Convert texts to sequences of integers"""
        sequences = []
        for text in texts:
            seq = []
            for word in text.lower().split():
                if word in self.word_index:
                    seq.append(self.word_index[word])
                else:
                    seq.append(self.word_index[self.oov_token])
            sequences.append(seq)
        return sequences
    
    def sequences_to_texts(self, sequences):
        """Convert sequences back to texts"""
        texts = []
        for seq in sequences:
            text = []
            for idx in seq:
                if idx in self.index_word:
                    text.append(self.index_word[idx])
                else:
                    text.append(self.oov_token)
            texts.append(' '.join(text))
        return texts

def pad_sequences(sequences, maxlen, padding='post', value=0):
    """Pad sequences to the same length"""
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded_seq = seq[:maxlen]
        else:
            if padding == 'post':
                padded_seq = seq + [value] * (maxlen - len(seq))
            else:  # 'pre'
                padded_seq = [value] * (maxlen - len(seq)) + seq
        padded_sequences.append(padded_seq)
    return padded_sequences

class AgriNLPDataset(Dataset):
    """Dataset for agricultural NLP model"""
    
    def __init__(self, prompts, responses, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Convert texts to sequences
        self.prompt_seqs = self.tokenizer.texts_to_sequences(self.prompts)
        self.response_seqs = self.tokenizer.texts_to_sequences(self.responses)
        
        # Add START and END tokens to responses
        for i in range(len(self.response_seqs)):
            self.response_seqs[i] = [self.tokenizer.word_index['<START>']] + self.response_seqs[i] + [self.tokenizer.word_index['<END>']]
        
        # Pad sequences
        self.prompt_seqs = pad_sequences(self.prompt_seqs, maxlen=self.max_length, padding='post')
        self.response_seqs = pad_sequences(self.response_seqs, maxlen=self.max_length, padding='post')
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        # For input, return the prompt sequence
        prompt = torch.tensor(self.prompt_seqs[idx], dtype=torch.long)
        
        # For target, we need the response sequence shifted by 1
        response_input = torch.tensor(self.response_seqs[idx][:-1], dtype=torch.long)
        response_target = torch.tensor(self.response_seqs[idx][1:], dtype=torch.long)
        
        # Pad if needed (should already be padded but just in case)
        if len(response_input) < self.max_length - 1:
            padding = torch.zeros(self.max_length - 1 - len(response_input), dtype=torch.long)
            response_input = torch.cat([response_input, padding])
            response_target = torch.cat([response_target, padding])
        
        return {
            'prompt': prompt,
            'response_input': response_input,
            'response_target': response_target
        }

class AgriNLPModel(nn.Module):
    """PyTorch model for agricultural NLP response generation"""
    
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(AgriNLPModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim * 2,  # * 2 because encoder is bidirectional
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim * 2, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, prompt, response=None):
        # Embedding
        prompt_embedded = self.embedding(prompt)
        
        # Encoding
        _, (hidden, cell) = self.encoder(prompt_embedded)
        
        # Convert encoder hidden state for decoder (bidirectional to unidirectional)
        hidden = hidden.view(NUM_LAYERS, 2, -1, HIDDEN_DIM)  # [layers, directions, batch, hidden]
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)  # [layers, batch, hidden*2]
        
        cell = cell.view(NUM_LAYERS, 2, -1, HIDDEN_DIM)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
        
        # If we have a response (training), use it for decoding
        if response is not None:
            response_embedded = self.embedding(response)
            decoder_output, _ = self.decoder(response_embedded, (hidden, cell))
            output = self.output(self.dropout(decoder_output))
            return output
        else:
            # For inference, we would implement beam search or greedy decoding here
            # Just a placeholder for now
            return None

def find_csv_files(path):
    """Recursively find all CSV files in a directory and its subdirectories"""
    csv_files = []
    
    # Check if the path is a directory
    if os.path.isdir(path):
        logging.info(f"Searching for CSV files in: {path}")
        
        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
    
    # If it's a single file and it's a CSV
    elif os.path.isfile(path) and path.endswith('.csv'):
        csv_files.append(path)
    
    logging.info(f"Found {len(csv_files)} CSV files")
    return csv_files

def load_dataset(path):
    """Load the dataset from a CSV file or directory structure"""
    try:
        # Find all CSV files in the path
        csv_files = find_csv_files(path)
        
        if not csv_files:
            logging.error(f"No CSV files found in {path}")
            return None
        
        # Initialize lists to store data
        all_prompts = []
        all_responses = []
        
        # Process each CSV file
        for file_path in tqdm(csv_files, desc="Loading CSV files"):
            try:
                logging.info(f"Processing: {file_path}")
                df = pd.read_csv(file_path)
                
                # Check for required columns
                if 'prompt' in df.columns and 'response' in df.columns:
                    prompts = df['prompt'].tolist()
                    responses = df['response'].tolist()
                    
                    logging.info(f"  Found {len(prompts)} prompt-response pairs")
                    all_prompts.extend(prompts)
                    all_responses.extend(responses)
                # Try alternative column names
                elif 'query' in df.columns and 'response' in df.columns:
                    prompts = df['query'].tolist()
                    responses = df['response'].tolist()
                    
                    logging.info(f"  Found {len(prompts)} query-response pairs")
                    all_prompts.extend(prompts)
                    all_responses.extend(responses)
                elif 'question' in df.columns and 'answer' in df.columns:
                    prompts = df['question'].tolist()
                    responses = df['answer'].tolist()
                    
                    logging.info(f"  Found {len(prompts)} question-answer pairs")
                    all_prompts.extend(prompts)
                    all_responses.extend(responses)
                else:
                    logging.warning(f"  Required columns not found in {file_path}")
                    logging.warning(f"  Available columns: {df.columns.tolist()}")
                    
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        
        if not all_prompts:
            logging.error("No valid prompt-response pairs found in any CSV file")
            return None
        
        logging.info(f"Loaded a total of {len(all_prompts)} prompt-response pairs")
        return all_prompts, all_responses
    
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None

def create_tokenizer(prompts, responses, max_words=10000):
    """Create and fit a tokenizer on all text"""
    # Combine prompts and responses for vocabulary
    all_texts = prompts + responses
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(num_words=max_words)
    tokenizer.fit_on_texts(all_texts)
    
    logging.info(f"Created tokenizer with {len(tokenizer.word_index)} words")
    return tokenizer

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        prompt = batch['prompt'].to(device)
        response_input = batch['response_input'].to(device)
        response_target = batch['response_target'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(prompt, response_input)
        
        # Reshape for cross entropy
        output = output.view(-1, output.size(2))
        response_target = response_target.view(-1)
        
        # Calculate loss (ignoring padding)
        loss = criterion(output, response_target)
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            prompt = batch['prompt'].to(device)
            response_input = batch['response_input'].to(device)
            response_target = batch['response_target'].to(device)
            
            # Forward pass
            output = model(prompt, response_input)
            
            # Reshape for cross entropy
            output = output.view(-1, output.size(2))
            response_target = response_target.view(-1)
            
            # Calculate loss
            loss = criterion(output, response_target)
            
            # Update statistics
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def generate_text(model, tokenizer, prompt, max_length=50, device="cpu"):
    """Generate text using the trained model"""
    model.eval()
    
    # Tokenize and pad prompt
    prompt_seq = tokenizer.texts_to_sequences([prompt.lower()])
    prompt_seq = pad_sequences(prompt_seq, maxlen=MAX_SEQ_LENGTH, padding='post')
    prompt_tensor = torch.tensor(prompt_seq, dtype=torch.long).to(device)
    
    # Encode the prompt
    prompt_embedded = model.embedding(prompt_tensor)
    _, (hidden, cell) = model.encoder(prompt_embedded)
    
    # Convert encoder hidden state for decoder
    hidden = hidden.view(NUM_LAYERS, 2, -1, HIDDEN_DIM)
    hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
    
    cell = cell.view(NUM_LAYERS, 2, -1, HIDDEN_DIM)
    cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
    
    # Start with START token
    current_token = torch.tensor([[tokenizer.word_index['<START>']]], dtype=torch.long).to(device)
    
    # Generate text
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # Embed current token
            token_embedded = model.embedding(current_token)
            
            # Decode one step
            output, (hidden, cell) = model.decoder(token_embedded, (hidden, cell))
            
            # Get prediction
            output = model.output(output)
            predicted_token = torch.argmax(output, dim=2).item()
            
            # Stop if we predict END token
            if predicted_token == tokenizer.word_index['<END>']:
                break
                
            # Add to generated tokens
            generated_tokens.append(predicted_token)
            
            # Update current token
            current_token = torch.tensor([[predicted_token]], dtype=torch.long).to(device)
    
    # Convert tokens to text
    generated_text = tokenizer.sequences_to_texts([generated_tokens])[0]
    
    return generated_text

def plot_training_history(history, save_path="nlp_training_history.png"):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train NLP model for agricultural responses")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file or directory with prompt-response pairs")
    parser.add_argument("--output", type=str, default=".", help="Output directory for model and tokenizer")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Maximum vocabulary size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set device
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"Using device: {device}")
    
    # Load dataset
    logging.info(f"Loading data from: {args.data}")
    dataset = load_dataset(args.data)
    if dataset is None:
        logging.error("Failed to load dataset. Exiting.")
        return
    
    prompts, responses = dataset
    
    # Create tokenizer
    tokenizer = create_tokenizer(prompts, responses, max_words=args.vocab_size)
    
    # Split into train and validation sets
    train_prompts, val_prompts, train_responses, val_responses = train_test_split(
        prompts, responses, test_size=0.1, random_state=42
    )
    
    # Create datasets
    train_dataset = AgriNLPDataset(train_prompts, train_responses, tokenizer)
    val_dataset = AgriNLPDataset(val_prompts, val_responses, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if device.type == "cuda" else 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 if device.type == "cuda" else 0
    )
    
    # Initialize model
    model = AgriNLPModel(
        vocab_size=len(tokenizer.word_index),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Training loop
    logging.info(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Log progress
        epoch_time = time.time() - epoch_start
        logging.info(f"Epoch {epoch+1}/{args.epochs} - {epoch_time:.2f}s - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Generate sample text
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            sample_prompt = val_prompts[0]
            sample_response = generate_text(model, tokenizer, sample_prompt, device=device)
            logging.info(f"Sample generation:")
            logging.info(f"Prompt: {sample_prompt}")
            logging.info(f"Generated: {sample_response}")
    
    # Calculate total training time
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f} seconds")
    
    # Save model and tokenizer
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "agric_nlp_model.pt")
    tokenizer_path = os.path.join(args.output, "tokenizer.pkl")
    
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    logging.info(f"Tokenizer saved to {tokenizer_path}")
    
    # Plot training history
    history_path = os.path.join(args.output, "nlp_training_history.png")
    plot_training_history(history, save_path=history_path)
    logging.info(f"Training history plot saved to {history_path}")

if __name__ == "__main__":
    main()