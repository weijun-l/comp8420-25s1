import torch
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_closest_tokens(embeddings, word_embedding_matrix):
    """
    Find the closest tokens in vocabulary for given embeddings using cosine similarity
    
    Args:
        embeddings: tensor of shape (batch_size, seq_len, hidden_size)
        word_embedding_matrix: model's word embedding matrix
    
    Returns:
        token_ids: tensor of reconstructed token IDs
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

def compute_gradient_difference(dummy_grad, origin_grad, distance_type="cosine"):
    """
    Compute the difference between dummy gradients and original gradients
    Uses COSINE DISTANCE by default (not L2!)
    
    Args:
        dummy_grad: gradients from dummy data
        origin_grad: gradients from original data  
        distance_type: "cosine" or "l2"
    
    Returns:
        grad_diff: scalar tensor representing gradient difference
    """
    grad_diff = 0
    
    if distance_type == "cosine":
        # Use cosine distance (1 - cosine_similarity)
        for dummy_g, origin_g in zip(dummy_grad, origin_grad):
            if dummy_g is not None and origin_g is not None:
                # Flatten gradients
                dummy_flat = dummy_g.view(-1)
                origin_flat = origin_g.view(-1)
                
                # Compute cosine similarity
                cos_sim = F.cosine_similarity(dummy_flat.unsqueeze(0), origin_flat.unsqueeze(0))
                # Convert to cosine distance (1 - cosine_similarity)
                grad_diff += (1 - cos_sim)
    else:
        # Fallback to L2 distance
        for dummy_g, origin_g in zip(dummy_grad, origin_grad):
            if dummy_g is not None and origin_g is not None:
                grad_diff += ((dummy_g - origin_g) ** 2).sum()
                
    return grad_diff

def text_gradient_leakage(model, origin_grad, true_label, tokenizer, input_length, 
                         num_iterations=150, init_size=1.4, coeff_reg=0.1):
    """
    Reconstruct text from gradients using Deep Leakage from Gradients method
    Uses improved optimization with regularization and COSINE DISTANCE
    
    Args:
        model: BERT model
        origin_grad: original gradients to match
        true_label: the actual label
        tokenizer: BERT tokenizer
        input_length: exact length of input sequence
        num_iterations: number of optimization iterations
        init_size: target norm for embedding regularization (default 1.4)
        coeff_reg: regularization coefficient (default 0.1)
    """
    device = next(model.parameters()).device
    
    # Initialize dummy embeddings with the same length as input
    dummy_embeds = torch.randn(
        1, input_length, model.config.hidden_size, 
        requires_grad=True, device=device
    )
    
    # Scale initial embeddings to target norm
    with torch.no_grad():
        current_norm = dummy_embeds.norm(p=2, dim=2).mean()
        dummy_embeds.data *= (init_size / current_norm)
    
    # Use known true label
    dummy_label = true_label.clone().detach().to(device)
    
    # Use Adam optimizer for better stability
    optimizer = torch.optim.Adam([dummy_embeds], lr=0.01)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Track best result
    best_final_error = None
    best_final_embeds = dummy_embeds.detach().clone()
    
    # print(f"Starting improved gradient-based text reconstruction for {input_length} tokens...")
    # print(f"Using COSINE DISTANCE, regularization coeff: {coeff_reg}, target norm: {init_size}")
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass with dummy embeddings
        outputs = model(inputs_embeds=dummy_embeds)
        logits = outputs.logits
        
        # Compute loss using known true label
        dummy_loss = criterion(logits, dummy_label)
        
        # Compute gradients with allow_unused=True
        dummy_grad = grad(dummy_loss, model.parameters(), create_graph=True, allow_unused=True)
        
        # Calculate reconstruction loss using COSINE DISTANCE (key improvement!)
        rec_loss = compute_gradient_difference(dummy_grad, origin_grad, distance_type="cosine")
        
        # Add regularization loss to keep embedding norms close to target
        reg_loss = (dummy_embeds.norm(p=2, dim=2).mean() - init_size).square()
        
        # Total loss
        tot_loss = rec_loss + coeff_reg * reg_loss
        
        # Backward pass and optimization step
        tot_loss.backward(retain_graph=True)
        optimizer.step()
        
        # Track best result
        with torch.no_grad():
            if best_final_error is None or tot_loss.item() <= best_final_error:
                best_final_error = tot_loss.item()
                best_final_embeds.data[:] = dummy_embeds.data[:]
        
        # Print progress every 50 iterations
        if iteration % 50 == 0:
            current_norm = dummy_embeds.norm(p=2, dim=2).mean().item()
            print(f"Iteration {iteration:3d}, Total Loss: {tot_loss.item():.6f}, Rec Loss: {rec_loss.item():.6f}, Reg Loss: {reg_loss.item():.6f}, Norm: {current_norm:.3f}")
            
            # Show current reconstruction
            word_embeddings = model.bert.embeddings.word_embeddings.weight
            reconstructed_ids = find_closest_tokens(dummy_embeds, word_embeddings)
            current_text = tokenizer.decode(reconstructed_ids[0], skip_special_tokens=True)
            print(f"Current reconstruction: {current_text}")
            print("-" * 60)
    
    # Use best result for final reconstruction
    print(f"\nUsing best result with error: {best_final_error:.6f}")
    word_embeddings = model.bert.embeddings.word_embeddings.weight
    reconstructed_ids = find_closest_tokens(best_final_embeds, word_embeddings)
    reconstructed_text = tokenizer.decode(reconstructed_ids[0], skip_special_tokens=True)
    
    return reconstructed_text, best_final_embeds