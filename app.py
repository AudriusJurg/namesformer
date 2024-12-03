import torch
import streamlit as st
from utils.sampler import sample_datasetless
import json
import random

# Lazy loading models and dictionaries
@st.cache
def load_models():
    return {
        "male": [torch.load(f"models/model_male_{name}.pt") for name in ["all", "partial", "top1000", "top500"]],
        "female": [torch.load(f"models/model_female_{name}.pt") for name in ["all", "partial", "top1000", "top500"]],
    }

@st.cache
def load_dictionaries(gender, mod_index):
    with open(f'dictionaries/{gender}_intchar_{mod_index}.json', 'r') as f:
        int_char = json.load(f)
    with open(f'dictionaries/{gender}_charint_{mod_index}.json', 'r') as f:
        char_int = json.load(f)
    return int_char, char_int

def validate_input(input_text, char_to_int):
    """
    Checks if all characters in input_text exist in char_to_int dictionary.
    Also checks that there are no whitespaces in the input.
    """
    return all(char in char_to_int and char != " " for char in input_text)

def generate_text(input_text, temperature, selected_model, gender, models, num_names, model_options, previous_outputs):
    mod_index = model_options.index(selected_model)
    model = models[gender][mod_index]
    int_char, char_int = load_dictionaries(gender, mod_index)

    if not validate_input(input_text, char_int):
        raise ValueError("Input contains characters not in the dictionary or whitespace!")

    outputs = []
    anti_infiniteloop_counter = 1
    while len(outputs) < num_names or anti_infiniteloop_counter > 1000:
        anti_infiniteloop_counter+=1
        # Generate a name
        new_name = sample_datasetless(model, int_to_char=int_char, char_to_int=char_int, start_str=input_text, temperature=temperature)

        # Strip whitespace and check that the name is not empty or a single letter
        new_name = new_name.strip()
        if new_name and len(new_name) > 1 and new_name not in previous_outputs:
            outputs.append(new_name)
            previous_outputs.add(new_name)  # Add to the set of previously generated names

    return outputs

# Streamlit UI
def main():
    st.title("Namesformer App")

    # Gender switch
    gender = st.radio("Select Gender:", ["Male", "Female"])

    # Model and temperature selection
    model_options = ["All", "Partial", "Popular_1000", "Popular_500"]
    selected_model = st.selectbox("Select a model", model_options)
    temperature = st.slider("Select temperature", 0.1, 2.0, 1.0, 0.1)

    # Number of names to generate
    num_names = st.selectbox("Number of names to generate:", [1, 3, 5, 10])

    models = load_models()

    # Set to track previously generated names (ensuring uniqueness)
    previous_outputs = set()

    # Input form
    with st.form(key="text_form"):
        input_text = st.text_input("Enter text:", value="", max_chars=2).lower()
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        try:
            # If input_text is empty, generate names with a random valid starting letter
            if not input_text:
                # Load the dictionary for the selected gender and model
                mod_index = model_options.index(selected_model)
                _, char_int = load_dictionaries(gender.lower(), mod_index)

                # Get the list of valid starting characters from the dictionary, excluding whitespace (index 0)
                valid_starting_chars = [char for char in char_int.keys() if char != " "]

                # Generate names with random starting letters
                outputs = []
                for _ in range(num_names):
                    random_start = random.choice(valid_starting_chars)
                    name_output = generate_text(random_start, temperature, selected_model, gender.lower(), models, 1, model_options, previous_outputs)
                    if name_output:  # Ensure the generated name is not empty
                        outputs.append(name_output[0])  # Since we're generating 1 name at a time

            else:
                # If input_text is provided, generate names starting with the provided input
                outputs = generate_text(input_text, temperature, selected_model, gender.lower(), models, num_names, model_options, previous_outputs)

            # Display results
            st.subheader(f"{gender} Names")
            for idx, output in enumerate(outputs, start=1):
                st.write(f"{idx}. {output.capitalize()}")

        except ValueError as e:
            st.error(str(e))

if __name__ == "__main__":
    main()