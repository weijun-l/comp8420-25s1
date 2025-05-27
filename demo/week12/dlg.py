import torch
import torch.nn.functional as F
from torch.autograd import grad
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_closest_tokens(embeddings, word_embedding_matrix):
    """
    Find the closest tokens in vocabulary for given embeddings using cosine similarity
    """
    batch_size, seq_len, hidden_size = embeddings.shape
    reconstructed_ids = []
    
    for batch_idx in range(batch_size):
        batch_ids = []
        for pos in range(seq_len):
            # Get current position embedding
            current_embed = embeddings[batch_idx, pos].detach().cpu().numpy().reshape(1, -1)
            vocab_embeds = word_embedding_matrix.detach().cpu().numpy()
            
            # Calculate cosine similarity with all vocab embeddings
            similarities = cosine_similarity(current_embed, vocab_embeds)[0]
            best_token_id = np.argmax(similarities)
            batch_ids.append(best_token_id)
        
        reconstructed_ids.append(batch_ids)
    
    return torch.tensor(reconstructed_ids)

def compute_gradient_difference(dummy_grad, origin_grad):
    """
    Compute the L2 difference between dummy gradients and original gradients
    """
    grad_diff = 0
    for dummy_g, origin_g in zip(dummy_grad, origin_grad):
        if dummy_g is not None and origin_g is not None:
            grad_diff += ((dummy_g - origin_g) ** 2).sum()
    return grad_diff

def text_gradient_leakage(model, origin_grad, true_label, tokenizer, input_length, num_iterations=150):
    """
    Reconstruct text from gradients using Deep Leakage from Gradients method
    Uses the exact same length as the input sentence (no padding)
    """
    device = next(model.parameters()).device
    
    # Initialize dummy embeddings with the same length as input
    dummy_embeds = torch.randn(
        1, input_length, model.config.hidden_size, 
        requires_grad=True, device=device
    )
    
    # Use known true label (no need to optimize it)
    dummy_label = true_label.clone().detach().to(device)
    
    # Use LBFGS optimizer (only optimize embeddings now)
    optimizer = torch.optim.LBFGS([dummy_embeds], lr=0.1)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Starting gradient-based text reconstruction for {input_length} tokens...")
    
    for iteration in range(num_iterations):
        def closure():
            optimizer.zero_grad()
            
            # Forward pass with dummy embeddings
            outputs = model(inputs_embeds=dummy_embeds)
            logits = outputs.logits
            
            # Compute loss using known true label
            dummy_loss = criterion(logits, dummy_label)
            
            # Compute gradients with allow_unused=True
            dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True, allow_unused=True)
            
            # Calculate gradient difference (this is what we minimize)
            grad_diff = compute_gradient_difference(dummy_grad, origin_grad)
            
            grad_diff.backward()
            return grad_diff
        
        # Optimization step
        loss = optimizer.step(closure)
        
        # Print progress every 50 iterations
        if iteration % 50 == 0:
            print(f"Iteration {iteration:3d}, Loss: {loss.item():.6f}")
            
            # Show current reconstruction
            word_embeddings = model.bert.embeddings.word_embeddings.weight
            reconstructed_ids = find_closest_tokens(dummy_embeds, word_embeddings)
            current_text = tokenizer.decode(reconstructed_ids[0], skip_special_tokens=True)
            print(f"Current reconstruction: {current_text}")
            print("-" * 50)
    
    # Final reconstruction
    word_embeddings = model.bert.embeddings.word_embeddings.weight
    reconstructed_ids = find_closest_tokens(dummy_embeds, word_embeddings)
    reconstructed_text = tokenizer.decode(reconstructed_ids[0], skip_special_tokens=True)
    
    return reconstructed_text, dummy_embeds