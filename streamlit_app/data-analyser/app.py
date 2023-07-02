import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TFRobertaForMaskedLM
import tensorflow as tf
def main():
    st.title("CSV Analysis App")
    st.write("Upload a CSV file and visualize the data.")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.subheader("Uploaded Data")
        st.write(data)
        correlation_matrix = data.corr()
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = TFRobertaForMaskedLM.from_pretrained("roberta-base")
        columns_string = ', '.join(data.columns)
        inputs = tokenizer("We want to make a scatter plot from our data and the columns are:" + columns_string +"the best columns from these to make scatter plots would be <mask> and <mask>", return_tensors="tf")
        logits = model(**inputs).logits

        # retrieve index of <mask>
        mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
        selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)

        predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
        sentence=tokenizer.decode(predicted_token_id)

        # Example: Get summary statistics using describe()
        summary_statistics = data.describe()
        constants = sentence.split()

        if len(constants) >= 2:
            best_columns_A = constants[0]
            best_columns_B = constants[1]
        else:
            best_columns_A = data.columns[1]
            best_columns_B = data.columns[2]
          # Perform data analysis and generate plots
        st.subheader("Data Analysis")

        # Summary statistics
        st.write("Summary Statistics:")
        st.write(data.describe())

        # Correlation heatmap
        st.write("Correlation Heatmap:")
        corr = data.corr()
       
        sns.heatmap(corr, annot=True)
        st.pyplot()

        
        # Histogram
        st.write("Histogram:")
        column = st.selectbox("Select a column for histogram", data.columns)
        plt.hist(data[column], bins=10)
        st.pyplot()

        # Box plot
        st.write("Box Plot:")
        column = st.selectbox("Select a column for box plot", data.columns)
        sns.boxplot(data=data, y=column)
        st.pyplot()

        # Scatter plot with regression line

        st.write("Scatter Plot of the best two columns chosen by AI:")
        st.write(best_columns_A + " and " + best_columns_B)
        sns.regplot(x=best_columns_A, y=best_columns_B, data=data)
        st.pyplot()

        

        # Count plot
        st.write("Count Plot:")
        column = st.selectbox("Select a column for count plot", data.columns)
        sns.countplot(data=data, x=column)
        st.pyplot()

        # Scatter plot matrix
        st.write("Scatter Plot Matrix:")
        sns.pairplot(data)
        st.pyplot()

if __name__ == '__main__':
    main()
