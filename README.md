# Sales Data Dashboard 📊

A comprehensive sales analytics dashboard built with Gradio, featuring interactive data visualization and real-time filtering capabilities.

## 🌟 Features

- **Interactive Dashboard**: Real-time sales data visualization with filtering
- **Multiple Chart Types**: Revenue trends, category breakdowns, and top products
- **Data Generation**: Synthetic sales data generator for testing and demos
- **Modern UI**: Clean, responsive Gradio interface
- **Performance Optimized**: Caching and efficient data processing
- **Comprehensive Logging**: Detailed application logging with Loguru

## 📁 Project Structure

```
gradio-dashboard/
├── sales_dashboard.py      # Main dashboard application
├── sales_data_generator.py # Synthetic data generator
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── sales_data.csv         # Generated sample data (created by generator)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/gradio-dashboard.git
   cd gradio-dashboard
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # Activate on Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   
   # Activate on macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Generate sample data** (first time only)
   ```bash
   python sales_data_generator.py
   ```
   This creates a `sales_data.csv` file with 100,000 synthetic sales records.

2. **Launch the dashboard**
   ```bash
   python sales_dashboard.py
   ```
   The dashboard will be available at `http://127.0.0.1:7860`

## 📊 Dashboard Features

### Key Metrics
- **Total Revenue**: Sum of all sales in the selected period
- **Total Orders**: Number of unique orders
- **Average Order Value**: Revenue per order
- **Top Category**: Best-performing product category

### Interactive Visualizations
1. **Revenue Over Time**: Line chart showing daily revenue trends
2. **Revenue by Category**: Bar chart of category performance
3. **Top Products**: Horizontal bar chart of best-selling products

### Filtering Options
- **Date Range**: Select start and end dates
- **Category Filter**: Focus on specific product categories
- **Real-time Updates**: Charts and metrics update instantly

## 🛠️ Technical Details

### Built With
- **[Gradio](https://gradio.app/)**: Web interface framework
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and analysis
- **[Polars](https://pola.rs/)**: High-performance data processing
- **[Matplotlib](https://matplotlib.org/)**: Data visualization
- **[Loguru](https://loguru.readthedocs.io/)**: Modern logging

### Architecture
- **Data Layer**: CSV-based storage with Pandas processing
- **Caching**: TTL cache for improved performance
- **UI Layer**: Gradio components with reactive updates
- **Logging**: Structured logging for debugging and monitoring

### Performance Features
- **Lazy Loading**: Data loaded only when needed
- **Caching**: Expensive operations cached with TTL
- **Efficient Filtering**: Optimized pandas operations
- **Memory Management**: Proper DataFrame copying and cleanup

## 📈 Data Schema

The generated sales data includes:

| Column | Type | Description |
|--------|------|-------------|
| `order_id` | Integer | Unique order identifier |
| `order_date` | Date | Order date (2023-2025 range) |
| `customer_id` | Integer | Customer identifier (100-999) |
| `customer_name` | String | Generated customer name |
| `product_id` | Integer | Product identifier |
| `product_names` | String | Product name |
| `categories` | String | Product category |
| `quantity` | Integer | Quantity ordered (1-10) |
| `price` | Float | Unit price ($1.99-$99.99) |
| `total` | Float | Total order value |

## 🔧 Customization

### Adding New Visualizations
1. Create data processing function in `sales_dashboard.py`
2. Add visualization function using matplotlib
3. Include in dashboard layout and update callbacks

### Modifying Data Generation
- Edit product lists in `sales_data_generator.py`
- Adjust date ranges, price ranges, or quantity limits
- Add new categories or customer segments

### Styling the Dashboard
- Modify CSS in the `gr.Blocks()` constructor
- Customize colors, fonts, and layout
- Add custom themes or branding

## 🐛 Troubleshooting

### Common Issues

**Missing data file**: Run `python sales_data_generator.py` first

**Port already in use**: Gradio will automatically find an available port

**Memory issues**: Reduce dataset size in the generator

**Date filtering errors**: Ensure proper date format (YYYY-MM-DD)

### Logging
The application uses Loguru for comprehensive logging:
```bash
# Check logs for detailed error information
# Logs appear in console with timestamps and log levels
```
---

**Built with ❤️ using Python and Gradio**
