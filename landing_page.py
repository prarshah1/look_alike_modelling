import streamlit as st
from streamlit import switch_page
import os
import sys
import pandas as pd

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Create a dictionary to store image data 
images = {
     "credit": {
        "path": "src/resources/ui_components/credit_cards.png",
        "title": "Credit Card Renewals",
        "description": "To retain credit card customers which might potentially churn away",
        "page_name": "pages/similance_credit.py"
    },
    "insurance": {
        "path": "src/resources/ui_components/insurance.png",
        "title": "Insurance budgeting",
        "description": "To budget for Insurance claims or Premiums",
        "page_name": "pages/similance_insurance.py"
    },
    "movie": {
        "path": "src/resources/ui_components/customer_acquisition.png",
        "title": "Customer Acquisition",
        "description": "Identify customers who might avail our cross selling offers in Theatre",
        "page_name": "pages/similance_movie.py"
    },
    "superstore": {
        "path": "src/resources/ui_components/shopping.png",
        "title": "CPG Retailers Campaign effectiveness",
        "description": "To make our Marketing Campaign more effective ",
        "page_name": "pages/similance_superstore.py"
    },
}

# Center align the title
st.markdown("<h1 style='text-align: center;'>Similance</h1>", unsafe_allow_html=True)

# Function to display images with detailed descriptions on separate page
def display_image_details(image_name, chosen_tab):
    selected_image_data = images[image_name]

    if chosen_tab == "details":
        st.subheader(selected_image_data["title"])
        st.image(selected_image_data["path"], width=300)  # Adjusted width for better presentation
        st.write(selected_image_data["description"])  # Display description

# Create a container to center the image gallery
center_container = st.container()
csv_paths = {
    "pages/similance_movie.py": "src/resources/data/movie_test.csv",
    "pages/similance_superstore.py": "src/resources/data/superstore_test.csv",
    "pages/similance_credit.py": "src/resources/data/credit_test.csv",
    "pages/similance_insurance.py": "src/resources/data/insurance_test.csv"
}
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
                    
            # Load CSV data from file and display the download button
            csv_file_path = csv_paths[images[image_name]["page_name"]]
            df = pd.read_csv(csv_file_path)
            csv_data = df.to_csv(index=False)
                
            st.download_button(
                label="Seed Data",
                data=csv_data,
                file_name=f"{image_name}_seed.csv",
                mime='csv'
            )

    # Display images and titles in the grid with boundary boxes and hover effects
    display_image_with_title(col1, "credit")
    st.write("")
    display_image_with_title(col2, "insurance")

# Create a new row for the second row of images
col3, col4 = st.columns(2)

# Display the remaining images and titles in the grid
display_image_with_title(col3, "movie")
st.write("")
display_image_with_title(col4, "superstore")

# Add shadow boxes around images
st.markdown("<style>img {box-shadow: 5px 5px 5px grey;}</style>", unsafe_allow_html=True)
