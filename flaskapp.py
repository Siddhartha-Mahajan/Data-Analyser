import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import io
import base64

app = Flask(__name__)

def perform_analysis(data):
    # Perform data analysis here
    # Replace this with your analysis code

    # Example: Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Example: Get summary statistics using describe()
    summary_statistics = data.describe()

    return correlation_matrix, summary_statistics

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
            correlation_matrix, summary_statistics = perform_analysis(data)

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

            # Return the correlation matrix, summary statistics, and heatmap image as JSON response
            return jsonify(correlation_matrix=correlation_matrix.to_dict(),
                           summary_statistics=summary_statistics.to_dict(),
                           heatmap_image=image_base64)

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
