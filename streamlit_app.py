import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from lightgbm import LGBMRegressor
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.preprocessing import RobustScaler


st.title("Seller Central Shipment Forecast")

excel_sht = st.file_uploader(label = "Upload SC Rolling Sales Report",
                             type = ".xlsx")

intro_tab, bto_tab, wto_tab = st.tabs(["Instructions", "BTO", "WTO"])

if excel_sht:
    SC_demand = pd.read_excel(excel_sht,
                            sheet_name = "Data")[["Week Ending", "SKU", "Units Ordered", "Units Ordered - B2B"]].rename({"Units Ordered":"Units Sold",
                                                                                                                        "Units Ordered - B2B": "Units Sold - B2B"},
                                                                                                                        axis=1)

    SC_demand["Week Ending"] = pd.to_datetime(SC_demand["Week Ending"])
    SC_demand["SKU"] = SC_demand["SKU"].replace({"40-05-WTO-A220-CS-stickerless":"40-05-WTO-A220-CS",
                                                "40-05-EVO-0750-CS-stickerless": "40-05-EVO-0750-CS"})
    SC_demand = SC_demand[SC_demand["SKU"].isin(["40-05-WTO-A220-CS", "40-05-BTO-A220-CS", "40-05-EVO-A712-CS", "40-05-EVO-0750-CS"])].reset_index(drop=True)

    SC_demand = SC_demand.groupby(["SKU",
                                pd.Grouper(key="Week Ending",
                                            freq = "W-SAT")])[["Units Sold", "Units Sold - B2B"]].sum().reset_index().drop_duplicates()
    
    # Create a MultiIndex for all SKU-week combinations
    sku_week_combinations = pd.MultiIndex.from_product([SC_demand["SKU"].unique(), 
                                                        pd.date_range(start=pd.to_datetime("2023-01-01"), 
                                                                      end=SC_demand["Week Ending"].max(), 
                                                                      freq='W-SAT')],
                                                       names=["SKU", "Week Ending"])

    # Create a skeleton DataFrame for all combinations
    skeleton_df = pd.DataFrame(index=sku_week_combinations)

    # Merge skeleton with your existing dataset (left join ensures all SKU-week combinations are kept)
    SC_demand_filled = pd.merge(skeleton_df, SC_demand, on=["SKU", "Week Ending"], how="left").set_index("Week Ending").fillna(0)

    ###############################
    # BEGIN BTO
    ###############################

    with bto_tab:

        if not excel_sht:
            st.warning("Please upload the Excel sheet before proceeding.")

        sc_sku = "40-05-BTO-A220-CS"
        last_date = SC_demand_filled.index.max()
        horizon = 20

        # LightGBM Forecaster
        LGBM_forec = ForecasterRecursive(regressor=LGBMRegressor(random_state=42, 
                                                                 max_depth=10, 
                                                                 n_estimators=300, 
                                                                 learning_rate=0.05, 
                                                                 verbose=-1),
                                        lags=16,
                                        window_features = RollingFeatures(stats=['mean', 'min', 'max'], window_sizes=8),
                                        transformer_y = RobustScaler()
        )

        LGBM_forec.fit(y=SC_demand_filled[SC_demand_filled["SKU"] == sc_sku].loc[:, "Units Sold"])

        prediction_df = pd.DataFrame({"Forecasted Units Sold": LGBM_forec.predict(horizon).reset_index(drop=True),
                                    "Week Ending": pd.date_range(start=last_date + pd.Timedelta(weeks=1), end=last_date + pd.Timedelta(weeks=horizon), freq="W-SAT")
                                }).set_index("Week Ending")

        prediction_df["forc_unit_csum"] = prediction_df["Forecasted Units Sold"].cumsum()
        prediction_df["Shipments"] = 0  

        st.header("BTO 20-Week Forecast")
        st.line_chart(
            SC_demand_filled[SC_demand_filled["SKU"] == sc_sku].merge(prediction_df, left_index=True, right_index=True, how="outer")[["Units Sold", "Forecasted Units Sold"]]
        )

        st.header("Shipment Planning")

        # User Inputs for Inventory
        bto_curr_inv = st.number_input(label="Current Inventory Level", 
                                       min_value=0, 
                                       key="BTO_Curr_Inv",
                                       value = 1000)
        bto_min_qty = st.number_input(label="Minimum Quantity Desired", 
                                      min_value=0, 
                                      key="BTO_Min_Qty", 
                                      value=960)
        prediction_df["Forecasted Inventory"] = np.clip(bto_curr_inv - prediction_df['forc_unit_csum'],0,None)

        # Update Inventory and Rerun for Chart Update
        if bto_curr_inv and bto_min_qty:

            # Ensure session state bto_shipments exist
            if "bto_shipments" not in st.session_state:
                st.session_state.bto_shipments = pd.DataFrame()

            st.write("Add Shipments:")
            shp_dt = st.date_input(label="Date of Shipment", 
                                    min_value=pd.to_datetime("today"), 
                                    max_value=prediction_df.index.max(), 
                                    key="BTO_shipment_date_input")

            shp_qty = st.number_input(label="Number of Units", 
                                        min_value=0, 
                                        key="BTO_shipment_qty_input")
            
            lead_time = st.number_input(label="Lead Time (Days)",
                                        min_value = 0,
                                        key = "BTO_Lead_Time")

            # Initialize Empty Shipments DataFrame
            shipment_df = pd.DataFrame()

            if st.button("Add Shipment", key = "BTO_Add_Shipment"):
                if shp_qty > 0:
                    new_shipment = pd.DataFrame([{"Week Ending": pd.to_datetime(shp_dt + pd.Timedelta(days=lead_time)), "Quantity": shp_qty}])
                    st.session_state.bto_shipments = pd.concat([st.session_state.bto_shipments, new_shipment], ignore_index=True)
                    st.success(f"Shipment of {shp_qty} units added for {pd.to_datetime(shp_dt + pd.Timedelta(days=lead_time))}.")

        # Update shipment values in prediction_df
                shipment_df = pd.DataFrame(st.session_state.bto_shipments).groupby(pd.Grouper(key = "Week Ending",
                                                                                      freq = "W-SAT"))["Quantity"].sum()

            if not shipment_df.empty:
                for date in shipment_df.index:
                    if date in prediction_df.index:
                        prediction_df.at[date, "Shipments"] += pd.DataFrame(shipment_df).at[date, "Quantity"]

                # Update Inventory
                updated_inventory = bto_curr_inv  # Start with initial inventory

                for i in range(len(prediction_df)):
                    updated_inventory -= prediction_df.iloc[i]["Forecasted Units Sold"]
                    updated_inventory += prediction_df.iloc[i]["Shipments"]
                    updated_inventory = max(0, updated_inventory)  # Ensure inventory doesn't go negative
                    prediction_df.at[prediction_df.index[i], "Forecasted Inventory"] = updated_inventory

            # Prepare Data for Altair Chart
            prediction_df_long = prediction_df.reset_index().melt(id_vars=["Week Ending"], 
                                                                value_vars=["Forecasted Units Sold", "Forecasted Inventory"], 
                                                                var_name="Metric", value_name="Value")

            # Create Line Chart
            chart = alt.Chart(prediction_df_long).mark_line().encode(
                    x="Week Ending:T",
                    y=alt.Y("Value:Q", title="Units"),
                    color=alt.Color("Metric:N", legend=alt.Legend(title="Legend", symbolSize=50, labelFontSize=12))
                )

            # Add Min Quantity Line
            min_qty_df = pd.DataFrame({"Week Ending": prediction_df_long["Week Ending"].unique(), "Value": bto_min_qty, "Metric": ["Min Quantity"] * len(prediction_df_long["Week Ending"].unique())})

            # Add Min Quantity Line
            min_qty_line = alt.Chart(min_qty_df).mark_line(color="red", strokeWidth=3).encode(
                x="Week Ending:T",
                y="Value:Q",
                color=alt.Color("Metric:N")
)
            
            # Combine Charts
            final_chart = (chart + min_qty_line).properties(width=900, height=400).interactive()
            st.altair_chart(final_chart, use_container_width=False)

            # DOWNLOADING DATA
            if not st.session_state.bto_shipments.empty:
                final_shipment_df = pd.DataFrame(st.session_state.bto_shipments)
                final_shipment_df["sku"] = sc_sku
                final_shipment_df["Week Ending"] = final_shipment_df["Week Ending"].dt.strftime('%Y-%m-%d')

                # Provide shipment download button
                csv = final_shipment_df.to_csv(index=False)
                st.download_button(
                        label="Download Shipments CSV, submit to pavlo@latourangelle.com",
                        data=csv,
                        file_name="shipments.csv",
                        mime="text/csv",
                        key = "BTO_Download_Shipments"
                )

    ################################
    # BEGIN WTO
    ################################
        with wto_tab:

            if not excel_sht:
                st.warning("Please upload the Excel sheet before proceeding.")

            sc_sku = "40-05-WTO-A220-CS"
            last_date = SC_demand_filled.index.max()
            horizon = 20

            # LightGBM Forecaster
            LGBM_forec = ForecasterRecursive(regressor=LGBMRegressor(random_state=42, 
                                                                    max_depth=10, 
                                                                    n_estimators=300, 
                                                                    learning_rate=0.05, 
                                                                    verbose=-1),
                                            lags=16,
                                            window_features = RollingFeatures(stats=['mean', 'min', 'max'], window_sizes=8),
                                            transformer_y = RobustScaler()
            )

            LGBM_forec.fit(y=SC_demand_filled[SC_demand_filled["SKU"] == sc_sku].loc[:, "Units Sold"])

            prediction_df = pd.DataFrame({"Forecasted Units Sold": LGBM_forec.predict(horizon).reset_index(drop=True),
                                        "Week Ending": pd.date_range(start=last_date + pd.Timedelta(weeks=1), end=last_date + pd.Timedelta(weeks=horizon), freq="W-SAT")
                                    }).set_index("Week Ending")

            prediction_df["forc_unit_csum"] = prediction_df["Forecasted Units Sold"].cumsum()
            prediction_df["Shipments"] = 0  

            st.header("WTO 20-Week Forecast")
            st.line_chart(
                SC_demand_filled[SC_demand_filled["SKU"] == sc_sku].merge(prediction_df, left_index=True, right_index=True, how="outer")[["Units Sold", "Forecasted Units Sold"]]
            )

            st.header("Shipment Planning")

            # User Inputs for Inventory
            wto_curr_inv = st.number_input(label="Current Inventory Level", 
                                        min_value=0, 
                                        key="WTO_Curr_Inv",
                                        value = 1000)
            wto_min_qty = st.number_input(label="Minimum Quantity Desired", 
                                        min_value=0, 
                                        key="WTO_Min_Qty", 
                                        value=960)
            prediction_df["Forecasted Inventory"] = np.clip(wto_curr_inv - prediction_df['forc_unit_csum'],0,None)

            # Update Inventory and Rerun for Chart Update
            if wto_curr_inv and wto_min_qty:

                # Ensure session state bto_shipments exist
                if "wto_shipments" not in st.session_state:
                    st.session_state.wto_shipments = pd.DataFrame()

                st.write("Add Shipments:")
                shp_dt = st.date_input(label="Date of Shipment", 
                                        min_value=pd.to_datetime("today"), 
                                        max_value=prediction_df.index.max(), 
                                        key="WTO_shipment_date_input")

                shp_qty = st.number_input(label="Number of Units", 
                                            min_value=0, 
                                            key="WTO_shipment_qty_input")
                
                lead_time = st.number_input(label="Lead Time (Days)",
                                            min_value = 0,
                                            key = "WTO_Lead_Time")

                # Initialize Empty Shipments DataFrame
                shipment_df = pd.DataFrame()

                if st.button("Add Shipment", key = "WTO_Add_Shipment"):
                    if shp_qty > 0:
                        new_shipment = pd.DataFrame([{"Week Ending": pd.to_datetime(shp_dt + pd.TimeDelta(lead_time)), "Quantity": shp_qty}])
                        st.session_state.wto_shipments = pd.concat([st.session_state.wto_shipments, new_shipment], ignore_index=True)
                        st.success(f"Shipment of {shp_qty} units added for {pd.to_datetime(shp_dt + pd.TimeDelta(lead_time))}.")

            # Update shipment values in prediction_df
                    shipment_df = pd.DataFrame(st.session_state.wto_shipments).groupby(pd.Grouper(key = "Week Ending",
                                                                                        freq = "W-SAT"))["Quantity"].sum()

                if not shipment_df.empty:
                    for date in shipment_df.index:
                        if date in prediction_df.index:
                            prediction_df.at[date, "Shipments"] += pd.DataFrame(shipment_df).at[date, "Quantity"]

                    # Update Inventory
                    updated_inventory = wto_curr_inv  # Start with initial inventory

                    for i in range(len(prediction_df)):
                        updated_inventory -= prediction_df.iloc[i]["Forecasted Units Sold"]
                        updated_inventory += prediction_df.iloc[i]["Shipments"]
                        updated_inventory = max(0, updated_inventory)  # Ensure inventory doesn't go negative
                        prediction_df.at[prediction_df.index[i], "Forecasted Inventory"] = updated_inventory

                # Prepare Data for Altair Chart
                prediction_df_long = prediction_df.reset_index().melt(id_vars=["Week Ending"], 
                                                                    value_vars=["Forecasted Units Sold", "Forecasted Inventory"], 
                                                                    var_name="Metric", value_name="Value")

                # Create Line Chart
                chart = alt.Chart(prediction_df_long).mark_line().encode(
                    x="Week Ending:T",
                    y=alt.Y("Value:Q", title="Units"),
                    color=alt.Color("Metric:N", legend=alt.Legend(title="Legend", symbolSize=50, labelFontSize=12))
                )

                # Add Min Quantity Line
                min_qty_df = pd.DataFrame({"Week Ending": prediction_df_long["Week Ending"].unique(), "Value": wto_min_qty, "Metric": ["Min Quantity"] * len(prediction_df_long["Week Ending"].unique())})

                # Add Min Quantity Line
                min_qty_line = alt.Chart(min_qty_df).mark_line(color="red", strokeWidth=3).encode(
                     x="Week Ending:T",
                    y="Value:Q",
                    color=alt.Color("Metric:N")
)

                # Combine Charts
                final_chart = (chart + min_qty_line).properties(width=900, height=400).interactive()
                st.altair_chart(final_chart, use_container_width=False)

                # DOWNLOADING DATA
                if not st.session_state.wto_shipments.empty:
                    final_shipment_df = pd.DataFrame(st.session_state.bto_shipments)
                    final_shipment_df["sku"] = sc_sku
                    final_shipment_df["Week Ending"] = final_shipment_df["Week Ending"].dt.strftime('%Y-%m-%d')

                    # Provide shipment download button
                    csv = final_shipment_df.to_csv(index=False)
                    st.download_button(
                            label="Download Shipments CSV, submit to pavlo@latourangelle.com",
                            data=csv,
                            file_name="shipments.csv",
                            mime="text/csv",
                            key = "WTO_Download_Shipment"
                    )

with intro_tab:

    st.subheader("Welcome! This tool will help us plan Amazon SC Shipments into the future.")

    st.markdown("""
                How to use it:
                - Click "Browse Files" above and upload Amazon Seller Central Report Rolling from teams.

                - Select any of the product tabs above to get started:
                    - At the top will be a 20-week demand forecast. 
                    - To view forecasted inventory, input the **current inventory level** (Found in Safety Stock Report), and the **minimum quantity desired** for this product.
                    - Examine the **Shipment Planning Chart** and use the *Add Shipment* feature to plan accordingly.
                    - Once all your shipments are planned, click "Download Shipments CSV" and submit to Pavlo.
                    - If you'd like to redo your shipment planning, click **Clear Session State** at the bottom of this page.
                
                Happy forecasting!
                """)

    st.divider()
    st.markdown("""
             Here's how it works: 
             - We use LightGBM to forecast product-level demand.
             - Using this demand forecast, we create an inventory forecast by 
                cumulatively subtracting the starting inventory from the forecasted demand.
             - We create and add planned shipments and update the rolling inventory forecast.
             - We save the shipment data in a csv and download.""")
    
    st.divider()
    st.write("RESTART SESSION HERE:")

    clear_state = st.button(label = "Clear Session State")

    if clear_state:
        st.session_state.clear()
