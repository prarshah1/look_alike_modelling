import streamlit as st
from streamlit import switch_page
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Create a dictionary to store image data 
images = {
    "Image 1": {
        "path": "src/resources/ui_components/customer_acquisition.png",
        "title": "Customer Acquisition",
        "description": "Identify customers who might avail our cross selling offers in Theatre",
        "page_name": "pages/similance_insurance.py"
    },
    "Image 2": {
        "path": "src/resources/ui_components/shopping.png",
        "title": "CPG Retailers Campaign effectiveness",
        "description": "To make our Marketing Campaign more effective ",
        "page_name": "similance_insurance"
    },
    "Image 3": {
        "path": "src/resources/ui_components/credit_cards.png",
        "title": "Credit Card Renewals",
        "description": "To retain credit card customers which might potentially churn away",
        "page_name": "similance_insurance"
    },
    "Image 4": {
        "path": "src/resources/ui_components/insurance.png",
        "title": "Insurance budgeting",
        "description": "To budget for Insurance claims or Premiums",
        "page_name": "similance_insurance"
    }
}

# Center align the title
st.markdown("<h1 style='text-align: center;'>Demo Stories</h1>", unsafe_allow_html=True)

# Function to display images with detailed descriptions on separate page
def display_image_details(image_name, chosen_tab):
    selected_image_data = images[image_name]

    if chosen_tab == "details":
        st.subheader(selected_image_data["title"])
        st.image(selected_image_data["path"], width=300)  # Adjusted width for better presentation
        st.write(selected_image_data["description"])  # Display description

# Create a container to center the image gallery
center_container = st.container()

# Create a 2x2 grid of columns within the centered container
with center_container:
    col1, col2 = st.columns(2)

    # Function to display images with boundary boxes and hover effects
    def display_image_with_title(image_col, image_name):
        with image_col:
            st.image(images[image_name]["path"], use_column_width=True, output_format='auto')
            button_clicked = st.button(images[image_name]["title"])
            if button_clicked:
                page_name = images[image_name]["page_name"]
                switch_page(page_name)

    # Display images and titles in the grid with boundary boxes and hover effects
    display_image_with_title(col1, "Image 1")
    st.write("")
    display_image_with_title(col2, "Image 2")

# Create a new row for the second row of images
col3, col4 = st.columns(2)

# Display the remaining images and titles in the grid
display_image_with_title(col3, "Image 3")
st.write("")
display_image_with_title(col4, "Image 4")

# Add shadow boxes around images
st.markdown("<style>img {box-shadow: 5px 5px 5px grey;}</style>", unsafe_allow_html=True)
