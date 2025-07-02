# -*- coding: utf-8 -*-
"""
sales_data_generator.py

This script generates a CSV file with 100,000 rows of sales data.
It then loads the CSV into a Polars DataFrame and pushes it to the Hugging Face hub.

"""

# import libraries
import polars as pl
import numpy as np
 #from datasets import load_dataset
import huggingface_hub as hf

from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

# Get the directory where the script is located
root_dir = Path(__file__).parent

# Construct the path to the CSV file in the root directory
csv_path = root_dir / "sales_data.csv"

def generate(nrows: int, filename: str):
    """
    Generate a synthetic sales dataset and write it to a CSV file.

    Parameters
    ----------
    nrows : int
        Number of rows (sales records) to generate.
    filename : str
        Path to the output CSV file.

    The generated dataset includes the following columns:
        - order_id: Unique order identifier.
        - order_date: Random date between 2023-01-01 and 2025-12-31.
        - customer_id: Random customer ID between 100 and 999.
        - customer_name: Randomly generated customer name.
        - product_id: Product identifier (offset by 200).
        - product_names: Name of the product.
        - categories: Product category.
        - quantity: Quantity of product ordered (1-10).
        - price: Price per unit (random float between 1.99 and 99.99).
        - total: Total price for the order (quantity * price).
    """
    names = np.asarray(
        [
            "Laptop",
            "Smartphone",
            "Desk",
            "Chair",
            "Monitor",
            "Printer",
            "Paper",
            "Pen",
            "Notebook",
            "Coffee Maker",
            "Cabinet",
            "Plastic Cups",
        ]
    )
    categories = np.asarray(
        [
            "Electronics",
            "Electronics",
            "Office",
            "Office",
            "Electronics",
            "Electronics",
            "Stationery",
            "Stationery",
            "Stationery",
            "Electronics",
            "Office",
            "Sundry",
        ]
    )
    product_id = np.random.randint(len(names), size=nrows)
    quantity = np.random.randint(1, 11, size=nrows)
    price = np.random.randint(199, 10000, size=nrows) / 100
    # Generate random dates between 2023-01-01 and 2025-12-31
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range = (end_date - start_date).days
    # Create random dates as np.array and convert to string format
    order_dates = np.array([
        (start_date + timedelta(days=np.random.randint(0, date_range))).strftime('%Y-%m-%d')
        for _ in range(nrows)
    ])
    # Define columns
    columns = {
        "order_id": np.arange(nrows),
        "order_date": order_dates,
        "customer_id": np.random.randint(100, 1000, size=nrows),
        "customer_name": [f"Customer_{i}" for i in np.random.randint(2**15, size=nrows)],
        "product_id": product_id + 200,
        "product_names": names[product_id],
        "categories": categories[product_id],
        "quantity": quantity,
        "price": price,
        "total": price * quantity,
    }
    # Create Polars DataFrame and write to CSV with explicit delimiter
    logger.info(f"Creating DataFrame with {nrows} rows")
    df = pl.DataFrame(columns)
    logger.info(f"Writing data to {filename}")
    df.write_csv(filename, separator=',', include_header=True)  # Ensure comma is used as the delimiter
    logger.success(f"Successfully generated {nrows} rows of sales data to {filename}")

# main program execution
if __name__ == "__main__":
    logger.info("Starting sales data generation script")
    
    # create the csv by Generating 100,000 rows of data with random order_date and save to CSV
    logger.info("Beginning data generation process...")
    generate(100000, str(csv_path))
    
    # load the csv into a polars dataframe for analysis
    logger.info("Loading generated data for verification...")
    df = pl.read_csv(str(csv_path))
    logger.info(f"Data shape: {df.shape}")
    print(df.head())
    logger.info("Sample data displayed above")
    
    # push the csv to huggingface datasets
    #hf.push_to_hub("ctgadget/generated_sales_data")
    logger.info("Script execution completed successfully")


