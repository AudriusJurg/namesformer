import torch
import streamlit as st
from utils.sampler import sample_datasetless
from components.classes import NameDataset, MinimalTransformer
import json

model_male_all=torch.load("models/model_male_all.pt")
model_female_all=torch.load("models/model_female_all.pt")

model_male_partial=torch.load("models/model_male_partial.pt")
model_female_partial=torch.load("models/model_female_partial.pt")

model_male_top1000=torch.load("models/model_male_top1000.pt")
model_female_top1000=torch.load("models/model_female_top1000.pt")

model_male_top500=torch.load("models/model_male_top500.pt")
model_female_top500=torch.load("models/model_female_top500.pt")

model_options = ["All","Partial","Popular_1000","Popular_500"]
model_options_male = [model_male_all,model_male_partial,model_male_top1000,model_male_top500]
model_options_female = [model_female_all,model_female_partial,model_female_top1000,model_female_top500]

def generate_text_with_model1(input_text,temperature,selected_model):
    mod_index = model_options.index(selected_model)
    model = model_options_male[mod_index]
    with open(f'dictionaries/male_intchar_{mod_index}.json', 'r') as f:
        int_char = json.load(f)
    with open(f'dictionaries/male_charint_{mod_index}.json', 'r') as f:
        char_int = json.load(f)
    return sample_datasetless(model,int_to_char=int_char,char_to_int=char_int,start_str=input_text,temperature=temperature)

def generate_text_with_model2(input_text,temperature,selected_model):
    mod_index = model_options.index(selected_model)
    model = model_options_female[mod_index]
    with open(f'dictionaries/female_intchar_{mod_index}.json', 'r') as f:
        int_char = json.load(f)
    with open(f'dictionaries/female_charint_{mod_index}.json', 'r') as f:
        char_int = json.load(f)
    return sample_datasetless(model,int_to_char=int_char,char_to_int=char_int,start_str=input_text,temperature=temperature)

# Streamlit interface
def main():
    # Set the title of the app
    st.title("Namesformer App")

    # Dropdown menu for selecting a model (you can replace with actual models)
    selected_model = st.selectbox("Select a model", model_options)

    temperature = st.slider("Select temperature", 0.1, 2.0, 1.0, 0.1)
    # Text input for the user to enter letters
    with st.form(key="text_form"):
        input_text = st.text_input("Enter text:").lower()
        submit_button = st.form_submit_button(label="Submit")

    if input_text:
        # Generate outputs for both genders
        male_output = generate_text_with_model1(input_text,temperature,selected_model)
        female_output = generate_text_with_model2(input_text,temperature,selected_model)

        # Display the outputs in two sections
        st.subheader("Male")
        st.write(male_output.capitalize())

        st.subheader("Female")
        st.write(female_output.capitalize())

if __name__ == "__main__":
    main()