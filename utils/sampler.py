import torch
def sample(model, dataset, start_str='a', max_length=20, temperature=1.0):
    assert temperature > 0, "Temperature must be greater than 0"
    model.eval()  # Switch model to evaluation mode
    with torch.no_grad():
        # Convert start string to tensor
        chars = [dataset.char_to_int[c] for c in start_str]
        input_seq = torch.tensor(chars).unsqueeze(0)  # Add batch dimension
        
        output_name = start_str
        for _ in range(max_length - len(start_str)):
            output = model(input_seq)
            
            # Apply temperature scaling
            logits = output[0, -1] / temperature
            probabilities = torch.softmax(logits, dim=0)
            
            # Sample a character from the probability distribution
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = dataset.int_to_char[next_char_idx]
            
            if next_char == ' ':  # Assume ' ' is your end-of-sequence character
                break
            
            output_name += next_char
            # Update the input sequence for the next iteration
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]])], dim=1)
        
        return output_name
    
def sample_datasetless(model, char_to_int, int_to_char, start_str='a', max_length=20, temperature=1.0):
    assert temperature > 0, "Temperature must be greater than 0"
    model.eval()  # Switch model to evaluation mode
    with torch.no_grad():
        # Convert start string to tensor
        chars = [char_to_int[c] for c in start_str]
        input_seq = torch.tensor(chars).unsqueeze(0)  # Add batch dimension
        
        output_name = start_str
        for _ in range(max_length - len(start_str)):
            output = model(input_seq)
            
            # Apply temperature scaling
            logits = output[0, -1] / temperature
            probabilities = torch.softmax(logits, dim=0)
            
            # Sample a character from the probability distribution
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = int_to_char[str(next_char_idx)]
            
            if next_char == ' ':  # Assume ' ' is your end-of-sequence character
                break
            
            output_name += next_char
            # Update the input sequence for the next iteration
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]])], dim=1)
        
        return output_name