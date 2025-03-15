import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from lightgbm import LGBMRegressor
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster
from sklearn.preprocessing import RobustScaler


st.title("Seller Central Shipment Forecast")

excel_sht = st.file_uploader(label = "Upload SC Rolling Sales Report",
                             type = ".xlsx")

intro_tab, bto_tab, wto_tab = st.tabs(["Instructions", "BTO", "WTO"])

with intro_tab:

    st.subheader("Welcome! This tool will help us plan Amazon SC Shipments.")

    st.markdown("""
                How to use it:
                - Click "Browse Files" above and upload Amazon Seller Central Report Rolling from teams.

                - Select any of the product tabs above to get started:
                    - At the top will be a 20-week demand forecast. 
                    - To view forecasted inventory, input the **current inventory level** (Found in Safety Stock Report).
                    - To use the shipment planning tool, you have two options: **Manual Shipment** or **Auto-Shipment**
                    - To use the **Manual Shipment Tool:**
                        - Examine the **Shipment Planning Chart** and use the *Add Shipment* feature to plan accordingly.
                    - To use the **Auto Shipment Tool:**
                        - Input desired Weeks of Cover, examine recommended shipment dataframe and forecasted inventory chart.
                        - When happy with results, click "Approve Shipments" to save them to memory.
                    - Once all your shipments are planned and saved, click "Download Shipments CSV" and submit to Pavlo.
                    - If you'd like to redo your shipment planning, click **Clear Session State** at the bottom of this page.
                
                Happy forecasting!
                """)

    st.divider()
    st.markdown("""
             Here's how it works: 
             - We use LightGBM to forecast product-level demand.
             - Using this demand forecast, we create an inventory forecast by 
                cumulatively subtracting the starting inventory from the forecasted demand.
             - we use an OLS model to estimate total order processing time and add it to any shipment.
             - We create and add planned shipments and update the rolling inventory forecast.
             - In the auto-shipment feature, we use an EOQ model to determine the optimal quantity per shipment.""")
    
    st.divider()
    st.write("RESTART SESSION HERE:")

    clear_state = st.button(label = "Clear Session State")

    if clear_state:
        st.session_state.clear()
      
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

    # Initializing crossfold validation
    cv = TimeSeriesFold(steps = 10,
                     initial_train_size = 50,
                     refit = True,
                     differentiation = 1
                     )

    # Defining Forecasting Function
    def run_forecast(sku, data, cross_validation):
    
        sc_sku = sku
        last_date = data.index.max()
        horizon = 35
        y_data = data[data["SKU"]==sc_sku].loc[:,"Units Sold"]
        cv = cross_validation
    
        # LightGBM Forecaster
        LGBM_forec = ForecasterRecursive(regressor = LGBMRegressor(random_state = 42, 
                                                                   verbose=1,
                                                                   max_depth = 10,
                                                                   n_estimators = 300,
                                                                   boosting_type = "dart"),
                                         window_features = RollingFeatures(stats=['mean', 'min', 'max'], window_sizes=8),
                                         transformer_y = RobustScaler(),
                                         differentiation = 1,
                                         lags = 8)
                
        results_grid = grid_search_forecaster(forecaster = LGBM_forec,
                                              y = y_data.reset_index(drop=True),
                                              param_grid = {"learning_rate":[0.1, 0.05, 0.01]},
                                              cv = cv,
                                              metric = 'mean_squared_error',
                                              return_best = True,
                                              n_jobs = 'auto',
                                              verbose = False,
                                              show_progress = False
                                             )
    
        LGBM_forec.fit(y=y_data)
    
        prediction_df = pd.DataFrame({"Forecasted Units Sold": LGBM_forec.predict(horizon).reset_index(drop=True),
                                      "Week Ending": pd.date_range(start=last_date + pd.Timedelta(weeks=1), end=last_date + pd.Timedelta(weeks=horizon), freq="W-SAT")
                                     }).set_index("Week Ending")
    
        prediction_df["forc_unit_csum"] = prediction_df["Forecasted Units Sold"].cumsum()
        prediction_df["Shipments"] = 0  
    
        return prediction_df

    # Defining OLS processing time estimator
    def OLS_prcssng_tm(quantity, creation_date):
        mth = pd.to_datetime(creation_date).month
        prcssng_tm = 0.02*quantity + 1.6734*mth + 33

        return np.round(prcssng_tm)

    # Defining backward OLS processing time estimator
    def OLS_prcssng_tm_bckwrd(quantity, close_date):
        qtr = pd.to_datetime(close_date).quarter
        prcssng_tm = 0.02*quantity + 0.8828*qtr + 36
    
        return np.round(prcssng_tm)

    # Defining EOQ function
    def EOQ_func(date, demand, min_qty = 4*6):
        qtr = pd.to_datetime(date).quarter
        if qtr == 1:
            strg_rt = 0.84
        elif qtr == 2 or qtr == 3:
            strg_rt = 0.78
        elif qtr == 4:
            strg_rt = 2.4
    
        EOQ = np.sqrt(80*demand/(0.1*strg_rt))
    
        return max(min_qty, np.round(EOQ))
    
    # Defining shipment auto-recommendation
    def shipment_reco(predicted_demand_df, initial_inventory, case_qty, weeks_of_cover=12):
    
        recommended_shipments = []
        pred_df = predicted_demand_df.copy()  # Work on a copy to avoid modifying the original DataFrame
        pred_df["Forecasted Inventory"] = 0  # Reset inventory for recalculating
        current_inventory = initial_inventory
        case_qty = case_qty
        
        for i in range(len(pred_df)):
        
            # If we've already created a shipment (no need to calculate forecasted inventory)
            if recommended_shipments:
                current_inventory = pred_df.iloc[i]["Forecasted Inventory"]
            # If no shipments have been made and we need to initialize forecasted inventory
            else:
                current_inventory -= pred_df.iloc[i]["Forecasted Units Sold"]
                current_inventory = max(0, current_inventory)
                pred_df.loc[pred_df.index[i], "Forecasted Inventory"] = current_inventory
        
            # Find point where inventory runs out
            if current_inventory == 0:
                zero_date = pred_df.index[i]  # Save date when inventory = 0
                shp_avail_dt = zero_date - pd.Timedelta(weeks=weeks_of_cover)  # Adjust for Desired Weeks of Cover
        
                # Find optimal shipment quantity
                optim_qty = EOQ_func(date=zero_date, demand=pred_df["Forecasted Units Sold"][:20].sum(), min_qty=4*6)
        
                # Estimate processing lead time
                prcss_wks = np.round(OLS_prcssng_tm_bckwrd(quantity=optim_qty, close_date=shp_avail_dt) / 7)
                creation_date = shp_avail_dt - pd.Timedelta(weeks=prcss_wks)
        
                # Send shipment out next monday if recommended creation date is before today
                if creation_date < pd.Timestamp.today():
                    creation_date = pd.to_datetime(np.busday_offset(np.datetime64('today', 'D') , offsets = 0, roll="forward", weekmask='Mon'))
                    shp_avail_dt = creation_date + pd.Timedelta(weeks = np.round(OLS_prcssng_tm(quantity=optim_qty, creation_date=creation_date) / 7))
        
                # Append shipment recommendation
                recommended_shipments.append({"Creation Date": creation_date, "Units": optim_qty, "Cases": np.round(optim_qty/case_qty), "Availability Date":shp_avail_dt})
        
                if shp_avail_dt >= pred_df.index[0] and shp_avail_dt in pred_df.index:
                    future_index = pred_df.index.get_loc(shp_avail_dt)
                    
                    # Add shipment quantity to inventory
                    current_inventory_shp = optim_qty + pred_df.loc[pred_df.index[future_index], "Forecasted Inventory"]
                    pred_df.loc[pred_df.index[future_index], "Forecasted Inventory"] = current_inventory_shp
        
                    # Recalculate running inventory
                    for j in range(future_index+1, len(pred_df)):
                        current_inventory_shp -= pred_df.iloc[j]["Forecasted Units Sold"]
                        current_inventory_shp = max(0, current_inventory_shp)
                        pred_df.loc[pred_df.index[j], "Forecasted Inventory"] = current_inventory_shp
        
                else:
                    print(f"Please lower desired Weeks of Cover.")
                    break
    
        return pd.DataFrame(recommended_shipments), pred_df

    ###############################
    # BEGIN BTO
    ###############################

    with bto_tab:

        sc_sku = "40-05-BTO-A220-CS"
        last_date = SC_demand_filled.index.max()
        horizon = 20

        ##############################
        # BEGIN FORECASTING
    
        bto_pred_df = run_forecast(sku = sc_sku, data = SC_demand_filled, cross_validation = cv)

        ##############################
        # END FORECASTING

        
        st.header("BTO 20-Week Forecast")
        st.line_chart(
            SC_demand_filled[SC_demand_filled["SKU"] == sc_sku].merge(bto_pred_df[:20], left_index=True, right_index=True, how="outer")[["Units Sold", "Forecasted Units Sold"]]
        )

        #####################################
        # START SHIPMENT PLANNING
        
        @st.fragment
        def shipment_planning_bto(predicted_df):
        
            bto_pred_df = predicted_df
            st.header("Shipment Planning")
        
            # User Inputs for Inventory
            bto_curr_inv = st.number_input(label="Current Inventory Level", 
                                               min_value=0, 
                                               key="BTO_Curr_Inv",
                                               value = 1000)
        
            bto_pred_df["Forecasted Inventory"] = np.clip(bto_curr_inv - bto_pred_df['forc_unit_csum'],0,None)

################################
            # SPLIT TABS INTO AUTO-RECS

            bto_manual_shipment_tab, bto_auto_shipment_tab = st.tabs(["Manual Shipments", "Auto-Shipment"])

            with bto_manual_shipment_tab:
                # Update Inventory and Rerun for Chart Update
                if bto_curr_inv:
            
                        # Ensure session state bto_shipments exist
                    if "bto_shipments" not in st.session_state:
                        st.session_state.bto_shipments = pd.DataFrame()
            
                    st.write("Add Shipments:")
                    shp_dt = st.date_input(label="Date of Shipment", 
                                                min_value=pd.to_datetime("today"), 
                                                max_value=bto_pred_df.index.max(), 
                                                key="BTO_shipment_date_input")
            
                    shp_qty = st.number_input(label="Number of Units", 
                                                    min_value=0, 
                                                    key="BTO_shipment_qty_input")
                    if shp_qty:
                        estimated_prcss_tm =  OLS_prcssng_tm(quantity = shp_qty, creation_date = shp_dt)
                        st.write(f"Estimated processing time: {estimated_prcss_tm} days.")
    
            
                    # Initialize Empty Shipments DataFrame
                    shipment_df = pd.DataFrame()
            
                    if st.button("Add Shipment", key = "BTO_Add_Shipment"):
                        if shp_qty > 0:
                            new_shipment = pd.DataFrame([{"Creation Date": pd.to_datetime(shp_dt),"Week Ending: Processed": pd.to_datetime(shp_dt + pd.Timedelta(days=estimated_prcss_tm)), "Quantity": shp_qty}])
                            st.session_state.bto_shipments = pd.concat([st.session_state.bto_shipments, new_shipment], ignore_index=True)
                            st.success(f"Shipment of {shp_qty} units added for {pd.to_datetime(shp_dt)}.")
            
                    # Update shipment values in prediction_df
                        shipment_df = pd.DataFrame(st.session_state.bto_shipments).groupby(pd.Grouper(key = "Week Ending: Processed",
                                                                                                  freq = "W-SAT"))["Quantity"].sum()
            
                    if not shipment_df.empty:
                        for date in shipment_df.index:
                            if date in bto_pred_df.index:
                                bto_pred_df.at[date, "Shipments"] += pd.DataFrame(shipment_df).at[date, "Quantity"]
            
                        # Update Inventory
                        updated_inventory = bto_curr_inv  # Start with initial inventory
            
                        for i in range(len(bto_pred_df)):
                            updated_inventory -= bto_pred_df.iloc[i]["Forecasted Units Sold"]
                            updated_inventory += bto_pred_df.iloc[i]["Shipments"]
                            updated_inventory = max(0, updated_inventory)  # Ensure inventory doesn't go negative
                            bto_pred_df.at[bto_pred_df.index[i], "Forecasted Inventory"] = updated_inventory

                # Prepare Data for Altair Chart
                bto_pred_df_long = bto_pred_df.reset_index().melt(id_vars=["Week Ending"], 
                                                                      value_vars=["Forecasted Units Sold", "Forecasted Inventory"], 
                                                                      var_name="Metric", value_name="Value")
        
                # Create Line Chart
                chart = alt.Chart(bto_pred_df_long).mark_line().encode(
                                x="Week Ending:T",
                                y=alt.Y("Value:Q", title="Units"),
                                color=alt.Color("Metric:N", legend=alt.Legend(title="Legend", symbolSize=50, labelFontSize=12))
                            )
                final_chart = chart.properties(width=900, height=400).interactive()
                st.altair_chart(final_chart, use_container_width=False)

            with bto_auto_shipment_tab:
                weeks_cover = st.number_input(label = "Weeks of Cover",
                                              min_value = 0,
                                              max_value = 15,
                                              key = "bto_wks_cvr")

                auto_shp_rec_df, bto_auto_pred_df = shipment_reco(predicted_demand_df = bto_pred_df, initial_inventory = bto_curr_inv, weeks_of_cover=weeks_cover, case_qty=20)
                
                edited_shpmt_df = st.data_editor(auto_shp_rec_df)

                if st.button(label = "Approve Shipments",
                             key = "BTO_SHIPMENT_APPROVAL"):
                    st.session_state.bto_shipments = edited_shpmt_df

                bto_auto_pred_df_long = bto_auto_pred_df.reset_index().melt(id_vars=["Week Ending"], 
                                                                      value_vars=["Forecasted Units Sold", "Forecasted Inventory"], 
                                                                      var_name="Metric", value_name="Value")
        
                # Create Line Chart
                auto_chart = alt.Chart(bto_auto_pred_df_long).mark_line().encode(
                                x="Week Ending:T",
                                y=alt.Y("Value:Q", title="Units"),
                                color=alt.Color("Metric:N", legend=alt.Legend(title="Legend", symbolSize=50, labelFontSize=12))
                            )
                final_auto_chart = auto_chart.properties(width=900, height=400).interactive()
                st.altair_chart(final_auto_chart, use_container_width=False)

################################
            # END TABS INTO AUTO-RECS

        
        shipment_planning_bto(predicted_df = bto_pred_df)

        #####################################
        # END SHIPMENT PLANNING

        # DOWNLOADING DATA
        final_shipment_df = pd.DataFrame(st.session_state.bto_shipments)
        final_shipment_df["sku"] = sc_sku
        final_shipment_df["Creation Date"] = final_shipment_df["Creation Date"].dt.strftime('%Y-%m-%d')

            # Provide shipment download button
        csv = final_shipment_df.to_csv(index=False)
        st.download_button(
                    label="Download Shipments CSV, submit to pavlo@latourangelle.com",
                    data=csv,
                    file_name="shipments.csv",
                    mime="text/csv",
                    key = "BTO_Download_Shipments"
            )

####################################
    # START WTO
####################################
    
    with wto_tab:

        sc_sku = "40-05-WTO-A220-CS"
        last_date = SC_demand_filled.index.max()
        horizon = 20

        ##############################
        # BEGIN FORECASTING
    
        wto_pred_df = run_forecast(sku = sc_sku, data = SC_demand_filled, cross_validation = cv)

        ##############################
        # END FORECASTING

        
        st.header("WTO 20-Week Forecast")
        st.line_chart(
            SC_demand_filled[SC_demand_filled["SKU"] == sc_sku].merge(wto_pred_df[:20], left_index=True, right_index=True, how="outer")[["Units Sold", "Forecasted Units Sold"]]
        )

        #####################################
        # START SHIPMENT PLANNING
        
        @st.fragment
        def shipment_planning_wto(predicted_df):
        
            wto_pred_df = predicted_df
            st.header("Shipment Planning")
        
            # User Inputs for Inventory
            wto_curr_inv = st.number_input(label="Current Inventory Level", 
                                               min_value=0, 
                                               key="WTO_Curr_Inv",
                                               value = 1000)
        
            wto_pred_df["Forecasted Inventory"] = np.clip(wto_curr_inv - wto_pred_df['forc_unit_csum'],0,None)

################################
            # SPLIT TABS INTO AUTO-RECS

            wto_manual_shipment_tab, wto_auto_shipment_tab = st.tabs(["Manual Shipments", "Auto-Shipment"])

            with wto_manual_shipment_tab:
                # Update Inventory and Rerun for Chart Update
                if wto_curr_inv:
            
                        # Ensure session state wto_shipments exist
                    if "wto_shipments" not in st.session_state:
                        st.session_state.wto_shipments = pd.DataFrame()
            
                    st.write("Add Shipments:")
                    shp_dt = st.date_input(label="Date of Shipment", 
                                                min_value=pd.to_datetime("today"), 
                                                max_value=wto_pred_df.index.max(), 
                                                key="WTO_shipment_date_input")
            
                    shp_qty = st.number_input(label="Number of Units", 
                                                    min_value=0, 
                                                    key="WTO_shipment_qty_input")
                    if shp_qty:
                        estimated_prcss_tm =  OLS_prcssng_tm(quantity = shp_qty, creation_date = shp_dt)
                        st.write(f"Estimated processing time: {estimated_prcss_tm} days.")
    
            
                    # Initialize Empty Shipments DataFrame
                    shipment_df = pd.DataFrame()
            
                    if st.button("Add Shipment", key = "WTO_Add_Shipment"):
                        if shp_qty > 0:
                            new_shipment = pd.DataFrame([{"Creation Date": pd.to_datetime(shp_dt),"Week Ending: Processed": pd.to_datetime(shp_dt + pd.Timedelta(days=estimated_prcss_tm)), "Quantity": shp_qty}])
                            st.session_state.wto_shipments = pd.concat([st.session_state.wto_shipments, new_shipment], ignore_index=True)
                            st.success(f"Shipment of {shp_qty} units added for {pd.to_datetime(shp_dt)}.")
            
                    # Update shipment values in prediction_df
                        shipment_df = pd.DataFrame(st.session_state.wto_shipments).groupby(pd.Grouper(key = "Week Ending: Processed",
                                                                                                  freq = "W-SAT"))["Quantity"].sum()
            
                    if not shipment_df.empty:
                        for date in shipment_df.index:
                            if date in wto_pred_df.index:
                                wto_pred_df.at[date, "Shipments"] += pd.DataFrame(shipment_df).at[date, "Quantity"]
            
                        # Update Inventory
                        updated_inventory = wto_curr_inv  # Start with initial inventory
            
                        for i in range(len(wto_pred_df)):
                            updated_inventory -= wto_pred_df.iloc[i]["Forecasted Units Sold"]
                            updated_inventory += wto_pred_df.iloc[i]["Shipments"]
                            updated_inventory = max(0, updated_inventory)  # Ensure inventory doesn't go negative
                            wto_pred_df.at[wto_pred_df.index[i], "Forecasted Inventory"] = updated_inventory

                # Prepare Data for Altair Chart
                wto_pred_df_long = wto_pred_df.reset_index().melt(id_vars=["Week Ending"], 
                                                                      value_vars=["Forecasted Units Sold", "Forecasted Inventory"], 
                                                                      var_name="Metric", value_name="Value")
        
                # Create Line Chart
                chart = alt.Chart(wto_pred_df_long).mark_line().encode(
                                x="Week Ending:T",
                                y=alt.Y("Value:Q", title="Units"),
                                color=alt.Color("Metric:N", legend=alt.Legend(title="Legend", symbolSize=50, labelFontSize=12))
                            )
                final_chart = chart.properties(width=900, height=400).interactive()
                st.altair_chart(final_chart, use_container_width=False)

            with wto_auto_shipment_tab:
                weeks_cover = st.number_input(label = "Weeks of Cover",
                                              min_value = 0,
                                              max_value = 15,
                                              key = "wto_wks_cvr")

                auto_shp_rec_df, wto_auto_pred_df = shipment_reco(predicted_demand_df = wto_pred_df, initial_inventory = wto_curr_inv, weeks_of_cover=weeks_cover, case_qty=20)
                
                edited_shpmt_df = st.data_editor(auto_shp_rec_df)

                if st.button(label = "Approve Shipments",
                             key = "WTO_SHIPMENT_APPROVAL"):
                    st.session_state.wto_shipments = edited_shpmt_df
                    

                wto_auto_pred_df_long = wto_auto_pred_df.reset_index().melt(id_vars=["Week Ending"], 
                                                                      value_vars=["Forecasted Units Sold", "Forecasted Inventory"], 
                                                                      var_name="Metric", value_name="Value")
        
                # Create Line Chart
                auto_chart = alt.Chart(wto_auto_pred_df_long).mark_line().encode(
                                x="Week Ending:T",
                                y=alt.Y("Value:Q", title="Units"),
                                color=alt.Color("Metric:N", legend=alt.Legend(title="Legend", symbolSize=50, labelFontSize=12))
                            )
                final_auto_chart = auto_chart.properties(width=900, height=400).interactive()
                st.altair_chart(final_auto_chart, use_container_width=False)

################################
            # END TABS INTO AUTO-RECS

        
        shipment_planning_wto(predicted_df = wto_pred_df)

        #####################################
        # END SHIPMENT PLANNING

        # DOWNLOADING DATA
        if not st.session_state.wto_shipments.empty:
            final_shipment_df = pd.DataFrame(st.session_state.wto_shipments)
            final_shipment_df["sku"] = sc_sku
            final_shipment_df["Creation Date"] = final_shipment_df["Creation Date"].dt.strftime('%Y-%m-%d')

            # Provide shipment download button
            csv = final_shipment_df.to_csv(index=False)
            st.download_button(
                    label="Download Shipments CSV, submit to pavlo@latourangelle.com",
                    data=csv,
                    file_name="shipments.csv",
                    mime="text/csv",
                    key = "WTO_Download_Shipments"
            )
