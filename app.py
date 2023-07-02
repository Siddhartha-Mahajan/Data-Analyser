import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import io
import base64
from transformers import AutoTokenizer, TFRobertaForMaskedLM
import tensorflow as tf

app = Flask(__name__)


def perform_analysis(data):
    # Perform data analysis here
    # Replace this with your analysis code

    # Example: Calculate the correlation matrix
    correlation_matrix = data.corr()
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = TFRobertaForMaskedLM.from_pretrained("roberta-base")
    columns_string = ', '.join(data.columns)
    inputs = tokenizer("We want to make a scatter plot from our data and the columns are:" & columns_string &"the best columns from these to make scatter plots would be <mask> and <mask>", return_tensors="tf")
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
        A = ""
        B = ""

    

    return correlation_matrix, summary_statistics, best_columns_A , best_columns_B

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file uploaded'})

    if file:
        try:
            # Read the CSV file into a pandas DataFrame
            data = pd.read_csv(file)

            # Perform analysis on the data
            correlation_matrix, summary_statistics, best_columns_A , best_columns_B = perform_analysis(data)

            # Generate a correlation heatmap image
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')

            # Save the plot to a BytesIO object
            image = io.BytesIO()
            plt.savefig(image, format='png')
            image.seek(0)

            # Convert the image to base64 string
            image_base64 = base64.b64encode(image.read()).decode('utf-8')

            # Generate the scatter plot of the best columns A and B
            plt.figure(figsize=(8, 6))
            plt.scatter(data[best_columns_A], data[best_columns_B])
            plt.xlabel(best_columns_A)
            plt.ylabel(best_columns_B)
            plt.title('Scatter Plot of {} vs {}'.format(best_columns_A, best_columns_B))

            # Save the scatter plot to a BytesIO object
            scatter_plot_image = io.BytesIO()
            plt.savefig(scatter_plot_image, format='png')
            scatter_plot_image.seek(0)

            # Convert the scatter plot image to base64 string
            scatter_plot_image_base64 = base64.b64encode(scatter_plot_image.read()).decode('utf-8')

            # Return the correlation matrix, summary statistics, heatmap image, and scatter plot image as JSON response
            return jsonify(correlation_matrix=correlation_matrix.to_dict(),
                           summary_statistics=summary_statistics.to_dict(),
                           heatmap_image=image_base64,
                           scatter_plot_image_best=scatter_plot_image_base64)
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
