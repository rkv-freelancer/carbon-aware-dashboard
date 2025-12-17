from WattsOnAI.draw import draw_csv

# Set your CSV file path here
csv_file_path = 'Output/bart_small_centralus_monitor.csv'

# Run the Dash app
if __name__ == "__main__":
    # Create the Dash app
    draw_csv(csv_file_path)
    