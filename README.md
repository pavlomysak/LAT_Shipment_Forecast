# Amazon Seller Central Forecasting & Inventory Planning App
For use by La Tourangelle, Inc

## Overview

Managing inventory for Amazon can be a challenging task, especially when balancing two fulfillment models—Amazon Vendor Central (PO-driven) and Amazon Seller Central (self-managed stock replenishment). Unlike Vendor Central, where Amazon generates purchase orders, Seller Central requires proactive forecasting and shipment planning to prevent stockouts while avoiding excessive inventory costs.

This application was developed to solve the Amazon Seller Central forecasting challenge by integrating machine learning-driven demand forecasting, rolling inventory calculations, and an interactive shipment planning tool—allowing stakeholders to make data-driven decisions for optimal stock management.

## Business Challenge

Traditional demand forecasting and inventory management methods often fall short when dealing with:

- Fluctuating Amazon demand due to seasonality, promotions, and competitor activity.
- Balancing stock levels to avoid overstocking fees while ensuring availability.
- Lack of visibility into future needs, making manual shipment planning inefficient.
  
Given these complexities, a data-driven solution was needed to predict future sales trends, optimize inventory levels, and empower stakeholders to make informed shipment decisions without guesswork.

## Solution

This application provides a three-part solution to address these challenges:

### Machine Learning-Driven Forecasting:
- Uses LightGBM to forecast demand based on historical sales trends, seasonality, and other factors.
- Generates a 20-week rolling forecast to anticipate future demand.

### Dynamic Inventory Tracking:
- Tracks expected inventory levels week-over-week, incorporating both sales forecasts and planned shipments.
- Ensures real-time visibility into potential stockouts or overstock situations.

### Interactive Shipment Planning Tool:
- Enables stakeholders to manually input and adjust planned shipments.
- Automatically updates inventory projections based on planned shipments.
- Provides real-time data visualization for intuitive decision-making.


## Key Features
✔️ Forecast Amazon Seller Central demand using LightGBM
✔️ Simulate rolling inventory based on projected demand and shipments
✔️ Dynamically adjust shipments & instantly visualize impact on inventory levels
✔️ Prevent stockouts and overstocking with proactive planning
✔️ Built-in interactivity for non-technical stakeholders

## Impact & Business Value
By implementing this tool, I transformed a manual, reactive process into a proactive, data-driven strategy for Amazon Seller Central inventory management. This application enables the company to:

- Reduce stockouts, ensuring consistent availability on Amazon.
- Optimize inventory levels, minimizing excess stock and storage fees.
- Improve operational efficiency, reducing the manual effort in shipment planning.
- Make data-driven decisions, leading to better forecasting accuracy and profitability.

## Technical Stack
Frontend: Streamlit (for interactive UI)
Backend & Forecasting: Python, Pandas, NumPy, LightGBM
Visualization: Altair

