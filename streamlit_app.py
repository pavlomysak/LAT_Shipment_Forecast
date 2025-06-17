import streamlit as st
import pandas as pd
import numpy as np
import copy
import altair as alt
from lightgbm import LGBMRegressor
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster
from sklearn.preprocessing import RobustScaler


st.title("Seller Central Shipment Forecast")

if "shipments" not in st.session_state:
        st.session_state.shipments = pd.DataFrame()

if "predictions" not in st.session_state:
    st.session_state.predictions = pd.DataFrame()

excel_sht = st.file_uploader(label = "Upload SC Rolling Sales Report",
                             type = ".xlsx")
inv_csv = st.file_uploader(label = "Upload Inventory Data",
                             type = ".csv")

intro_tab, shpt_ovr_tab, shp_inspct_tab = st.tabs(["Instructions", "Shipment Overview", "Shipment Inspector"])

sku_data = {"40-05-BTO-A220-CS": ["1999-01-01", 20, 75],
            "40-05-WTO-A220-CS": ["1999-01-01", 20, 75],
            "40-05-EVO-0750-CS": ["2024-01-01", 6, 132],
            "40-05-HZL-0500-CS": ["2024-01-01", 6, 196],
            "40-05-WAL-50BIB-CS": ["2023-02-01", 3, 60],
            "40-05-AVO-50BIB-CS": ["2023-06-01", 3, 60],
            "40-05-TSO-50BIB-CS": ["2023-01-01", 3, 60],
            "40-05-GSO-50BIB-CS": ["2023-01-01", 3, 60],
            "40-05-EVO-50BIB-CS": ["2023-01-01", 3, 60],
            "40-05-PEA-50BIB-CS": ["2023-01-01", 3, 60],
            "40-05-OCA-50BIB-CS": ["2023-01-01", 3, 60]}
sku_replacement = {"40-05-WTO-A220-CS-stickerless":"40-05-WTO-A220-CS",
                   "40-05-EVO-0750-CS-stickerless": "40-05-EVO-0750-CS",
                   "40-05-EVO-A712-CS": "40-05-EVO-0750-CS",
                   "40-05-HZL-A512-CS":"40-05-HZL-0500-CS",
                   857190000675:"40-05-PIO-0250-CS",
                   "40-05-GSO-50BIB-CS-stickerless":"40-05-GSO-50BIB-CS"}

if inv_csv:
    inv_dt = pd.read_csv(inv_csv)
    inv_dt["sku"] = inv_dt["sku"].replace(sku_replacement)
    inv_cols_of_interest = ['afn-warehouse-quantity', 'afn-inbound-working-quantity', 'afn-inbound-shipped-quantity', 'afn-inbound-receiving-quantity', 'afn-unsellable-quantity']
    inv_dt = inv_dt.groupby("sku")[inv_cols_of_interest].sum()
    current_inventory_levels = inv_dt.iloc[:,:4].sum(axis=1) - inv_dt['afn-unsellable-quantity']

with intro_tab:

    st.subheader("Welcome!")

    st.markdown("""
### 🚚 How to Use This Tool

1. **Upload Your Files**  
   - Upload the **Seller Central Rolling Sales** report (from Teams).  
   - Upload the **Inventory Report** from the *Manage FBA Inventory* tab on Amazon Seller Central.

2. **Shipment Overview Tab**  
   - View **all recommended shipments** in one place.  
   - You can **sort shipments by creation date** by clicking the *Creation Date* column header.  
   - To start over, click **Run Shipment Recommendations** — but note this will **reset all saved shipment planning progress**.

3. **Shipment Inspector Tab**  
   - Click on any shipment to view detailed information:  
     - **20-week demand forecast**  
     - A visual of **planned shipments vs. demand**  
     - A plain-English explanation of the **shipment logic**
   - Want to adjust the settings (e.g., Weeks of Cover)?  
     - Tweak the values and click **Approve Shipments** to save changes.  
     - This will **update the master shipment list** (in the Overview tab) with only the approved shipments for that SKU.


**⚠️ Note:**  
Running **Run Shipment Recommendations** in the **Overview tab** will **erase all current shipment plans** and start from scratch. Be sure you're ready before proceeding.


**Happy forecasting!**

                """)

    st.divider()
    st.markdown("""
             Here's how it works: 
             - We use LightGBM to forecast product-level demand.
             - Using this demand forecast, we create an inventory forecast by 
                cumulatively subtracting the starting inventory from the forecasted demand.
             - we use an OLS model to estimate total order processing time and add it to any shipment.
             - We create and add planned shipments and update the rolling inventory forecast.
             - We use an EOQ model to determine the optimal quantity per shipment.""")
    
    st.divider()

      
if excel_sht:
    SC_demand = pd.read_excel(excel_sht,
                            sheet_name = "Data")[["Week Ending", "SKU", "Units Ordered", "Units Ordered - B2B"]].rename({"Units Ordered":"Units Sold",
                                                                                                                        "Units Ordered - B2B": "Units Sold - B2B"},
                                                                                                                        axis=1)

    SC_demand["Week Ending"] = pd.to_datetime(SC_demand["Week Ending"])
    SC_demand["SKU"] = SC_demand["SKU"].replace(sku_replacement)
    SC_demand = SC_demand[SC_demand["SKU"].isin(["40-05-WTO-A220-CS", "40-05-BTO-A220-CS", 
                                                 "40-05-EVO-A712-CS", "40-05-EVO-0750-CS", 
                                                 "40-05-HZL-0500-CS", "40-05-WAL-50BIB-CS", 
                                                 "40-05-AVO-50BIB-CS", "40-05-TSO-50BIB-CS", 
                                                 "40-05-GSO-50BIB-CS", "40-05-PEA-50BIB-CS", 
                                                 "40-05-EVO-50BIB-CS", "40-05-OCA-50BIB-CS"])].reset_index(drop=True)

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

    def run_forecast(data, cross_validation):
        prediction_dict = {}
        
        for sc_sku in sku_data.keys():
                
            last_date = data.index.max()
            horizon = 42
            y_data = data[(data["SKU"]==sc_sku)].loc[:,"Units Sold"]
            cv = cross_validation
            
                # LightGBM Forecaster
            LGBM_forec = ForecasterRecursive(regressor = LGBMRegressor(random_state = 42, 
                                                                           verbose=0,
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
            
            prediction_df = pd.DataFrame({"Forecasted Units Sold": round(LGBM_forec.predict(horizon).reset_index(drop=True)),
                                              "Week Ending": pd.date_range(start=last_date + pd.Timedelta(weeks=1), end=last_date + pd.Timedelta(weeks=horizon), freq="W-SAT")
                                             }).set_index("Week Ending")
            
            prediction_dict.update({sc_sku: np.clip(prediction_df, 0, None)})
    
        return prediction_dict
    

    # Defining OLS processing time estimator
    def OLS_prcssng_tm(quantity, creation_date):
        mth = pd.to_datetime(creation_date).month
        prcssng_tm = 0.011*quantity + 0.24*mth + 40.22

        return np.round(prcssng_tm)

    # Defining backward OLS processing time estimator
    def OLS_prcssng_tm_bckwrd(quantity, close_date):
        mth = pd.to_datetime(close_date).month
        prcssng_tm = 0.011*quantity + -0.52*mth + 45.42
        
        return np.round(prcssng_tm)

    # Defining EOQ function
    def EOQ_func(date, demand, sku, min_qty = 4*6):
        qtr = pd.to_datetime(date).quarter

        if sku[12:15]=="BIB":
            volume = 0.5
        if sku[11:14]=="750" or sku[11:14]=="712":
            volume = 0.044

        if sku[11:14]=="220":
            volume = 0.1

        if sku[11:14]=="500" or sku[11:14]=="512":
            volume = 0.031
        
        if qtr == 4: # If in Q4; $2.4
            strg_rt = 2.4
        else: # If in Q1, Q2, Q3; $0.78
            strg_rt = 0.78

        EOQ = np.sqrt(120*demand*6*1.2/(volume*strg_rt*6))
    
        return max(min_qty, np.round(EOQ))

    def generate_recommended_shipments(prediction_dict, weeks_of_cover_dict, recommended_shipments = None, explanations = None):
        
        predictions = prediction_dict.copy()

        if recommended_shipments == None:
            recommended_shipments = []
            explanations = {sku: [] for sku in predictions}

        forecast_horizon = len(next(iter(predictions.values()))["Forecasted Units Sold"])

        # Initialize forecasted inventory
        for sku in predictions:
            if "Forecasted Inventory" not in predictions[sku].columns:
                predictions[sku]["Forecasted Inventory"] = [0] * forecast_horizon
                predictions[sku]["Forecasted Inventory"][0] = current_inventory_levels[sku]

                for week_i in range(forecast_horizon):
                    if week_i > 0:
                        prev_inv = predictions[sku]["Forecasted Inventory"][week_i - 1]
                        units_sold = predictions[sku]["Forecasted Units Sold"][week_i - 1]
                        predictions[sku]["Forecasted Inventory"][week_i] = max(prev_inv - units_sold, 0)
        

        # BASE CASE ################ ---------------------------------------------------
        if all([value.iloc[-1,1]!=0 for value in predictions.values()]):
            recod_shipments = pd.DataFrame(recommended_shipments)
            cases = []
            pallets = []
            for row in range(len(recod_shipments)):
                case = round(recod_shipments["Units"][row]/sku_data[recod_shipments["SKU"][row]][1])
                pallet = round(case / sku_data[recod_shipments["SKU"][row]][2], 2)
                cases.append(case)
                pallets.append(pallet)
            recod_shipments["Cases"] = cases
            recod_shipments["Pallets"] = pallets

            return recod_shipments, predictions, explanations
        # BASE CASE ################ ---------------------------------------------------
        
        for week_i in range(forecast_horizon-1, 0, -1):
            print(f"--------------- NEW WEEK ------------------------------------ {week_i}")
            ttl_qty = 0
            wkly_shpmts = []
            for sku in predictions:
                print(f"------------- {sku} ----------------------------------------------")

                # Trigger shipment if inventory just dropped to 0
                if (predictions[sku]["Forecasted Inventory"][week_i:].sum() == 0) & (predictions[sku]["Forecasted Inventory"][week_i - 1] != 0):
                    print(f"TRIGGER SHIPMENT -- FUTURE INV SUM: {predictions[sku]['Forecasted Inventory'][week_i:].sum()}; last week inventory: {predictions[sku]['Forecasted Inventory'][week_i - 1]}")
                    zero_date = predictions[sku].index[week_i]
                    shp_avail_dt = zero_date - pd.Timedelta(weeks=weeks_of_cover_dict[sku])
                    forecasted_demand = sum(predictions[sku]["Forecasted Units Sold"][week_i:week_i + 4])
                    optim_qty = EOQ_func(date=zero_date, demand=forecasted_demand, sku=sku, min_qty=4 * 6)
                    optim_case_qty = round(optim_qty / sku_data[sku][1])
                    optim_pallet_qty = optim_case_qty / sku_data[sku][2]
                    
                    # if pallet +- 0.2, round to pallet qty
                    rounded_pallet_qty = round(optim_pallet_qty)
                    if abs(optim_pallet_qty - rounded_pallet_qty) <= 0.2 and rounded_pallet_qty >= 1:
                        optim_qty = int(rounded_pallet_qty * sku_data[sku][2] * sku_data[sku][1])

                    # Save SKU and Quantity information in weekly temp list
                    wkly_shpmts.append({"SKU": sku, "Quantity": optim_qty})
                    ttl_qty += optim_qty

                    
            print(f"TOTAL QUANTITY: {ttl_qty}")
            # Estimate processing lead time
            if ttl_qty != 0:
                prcss_wks = round(OLS_prcssng_tm_bckwrd(quantity=ttl_qty, close_date=shp_avail_dt) / 7)
                creation_date = shp_avail_dt - pd.Timedelta(weeks=prcss_wks)
                
                # If creation date is in the past, push it up
                while creation_date < pd.Timestamp.today():
                    
                    # push creation_date up 1 week
                    creation_date += pd.Timedelta(weeks = 1)
                    print(f"--- We have a recommendation for {creation_date} w avilability {shp_avail_dt}... but are there any other shipments happening today??")

                # check to see if there are any existing shipments on that week
                temp_reco_dt = pd.DataFrame(recommended_shipments, columns = ["SKU", "Creation Date", "Availability Date", "Units"])
                if creation_date in temp_reco_dt["Creation Date"].unique():

                    print(f"{sku} --- YES!! Updating quantity!!!")
                    ttl_qty += temp_reco_dt[temp_reco_dt["Creation Date"]==creation_date]["Units"].sum()
                    prcss_wks = round(OLS_prcssng_tm(quantity=ttl_qty, creation_date=creation_date) / 7)
                    shp_avail_dt = creation_date + pd.Timedelta(weeks=prcss_wks)

                    for shp in wkly_shpmts:
                        explanations[shp["SKU"]].append(
                            f"Added to existing shipment on {creation_date.date()}. \n"
                            f"Updated availability: {shp_avail_dt.date()}."
    )

                    # update old shipment availability dates
                    print(f"Other SKU's for this date: {temp_reco_dt[temp_reco_dt['Creation Date']==creation_date]['SKU'].unique()}")
                    for shipment in recommended_shipments:
                        if shipment["Creation Date"] == creation_date:
                            shipment["Availability Date"] = shp_avail_dt
                            explanations[shipment["SKU"]].append(
                                f"New SKU's added to shipment on {creation_date.date()}. \n"
                                f"Updated availability: {shp_avail_dt.date()}.")

                else:
                    print(f" --- Nope! Just will update shipment availability date")
                    prcss_wks = round(OLS_prcssng_tm(quantity=ttl_qty, creation_date=creation_date) / 7)
                    shp_avail_dt = creation_date + pd.Timedelta(weeks=prcss_wks)

            for shp_sku in wkly_shpmts:
                print(f"Adding Shipment: SKU: {shp_sku['SKU']}, Availability Date: {shp_avail_dt}, Creation Date: {creation_date}")
                explanations[shp_sku["SKU"]].append(
                            f"Projected OOS on {zero_date.date()}. \n"
                            f"Shipment created on {creation_date.date()} to arrive by {shp_avail_dt.date()} \n"
                            f"(estimated processing time: {prcss_wks} weeks)."
                        )
                recommended_shipments.append({
                            "SKU": shp_sku["SKU"],
                            "Creation Date": creation_date,
                            "Availability Date": shp_avail_dt,
                            "Units": shp_sku["Quantity"],
                        })


        ### Update forecasted inventory when shipment arrives
        for shipment in recommended_shipments:
            shp_sku = shipment["SKU"]
            shp_avail_dt = shipment["Availability Date"]
            shp_qty = shipment["Units"]

            if shp_avail_dt in predictions[shp_sku].index:
                avail_idx = predictions[shp_sku].index.get_loc(shp_avail_dt)
            else:
                continue
            
            predictions[shp_sku]["Forecasted Inventory"][avail_idx] += shp_qty

            ### Recalculate downstream inventory
            for j in range(avail_idx + 1, forecast_horizon):
                prev_inv = predictions[shp_sku]["Forecasted Inventory"][j - 1]
                units_sold = predictions[shp_sku]["Forecasted Units Sold"][j - 1]
                predictions[shp_sku]["Forecasted Inventory"][j] = max(prev_inv - units_sold, 0)

        print("RECURSING...")
        return generate_recommended_shipments(prediction_dict = predictions, weeks_of_cover_dict = weeks_of_cover_dict, recommended_shipments = recommended_shipments, explanations = explanations)



    def product_tab(sku, pred_df, case_qty, pallet_qty):
    
        st.header(f"{sku} 20-Week Forecast")
        st.line_chart(
            SC_demand_filled[SC_demand_filled["SKU"] == sku].merge(st.session_state.predictions[sku][:20], left_index=True, right_index=True, how="outer")[["Units Sold", "Forecasted Units Sold"]]
        )

        #####################################
        # START SHIPMENT PLANNING
        
        @st.fragment
        def shipment_planning_func(predicted_df):
        
            pred_df = predicted_df
            st.header("Shipment Planning")

################################
            # SPLIT TABS INTO AUTO-RECS

            #shipment_memory = pd.DataFrame()
            #weeks_cover = st.number_input(label = "Weeks of Cover",
            #                              min_value = 0, 
            #                              max_value = 15,
            #                              value = woc_dict[sku],
            #                              key = f"wks_cvr{sku}")
            
            #temp_woc_dict = {sku: woc for sku in list(sku_data.keys())}
            #temp_woc_dict[sku] = weeks_cover
            temp_logic = copy.deepcopy(st.session_state.logic)

            #auto_shp_rec_df, auto_pred_df = generate_recommended_shipments(pred_dict, temp_woc_dict)

            with st.expander("Shipment Logic"):

                temp_sku_logs = temp_logic.get(sku)
                    
                if temp_sku_logs:
                    st.markdown(f"### Logic for SKU: `{sku}`")
                    for log_entry in temp_sku_logs:
                        st.markdown(f"- {log_entry}")
                else:
                    st.info("No logs for this SKU.")

            auto_pred_df_long = st.session_state.predictions[sku].reset_index().melt(id_vars=["Week Ending"], 
                                                                value_vars=["Forecasted Units Sold", "Forecasted Inventory"], 
                                                                var_name="Metric", value_name="Value")

            # st.table(st.session_state.predictions[sku])
            # Create Line Chart
            auto_chart = alt.Chart(auto_pred_df_long).mark_line().encode(
                                x="Week Ending:T",
                                y=alt.Y("Value:Q", title="Units"),
                                color=alt.Color("Metric:N", legend=alt.Legend(title="Legend", symbolSize=50, labelFontSize=12))
                            )
            final_auto_chart = auto_chart.properties(width=900, height=400).interactive()
            st.altair_chart(final_auto_chart, use_container_width=False)

            #if st.button(label = "Approve and Update Shipments",
             #           key = f"AUTO_SHIPMENT_APPROVAL{sku}"):
             #   shipment_memory = auto_shp_rec_df
             #   shipment_memory["SKU"] = sku
             #   st.session_state.shipments = st.session_state.shipments[st.session_state.shipments["SKU"] != sku]
             #   st.session_state.shipments = pd.concat([st.session_state.shipments, shipment_memory])
             #   st.rerun()
             #   st.success(f"{sku} Shipments saved!")

        shipment_planning_func(predicted_df = pred_df)



    
    with shpt_ovr_tab:

        woc = st.number_input(label = "Weeks of Cover",
                       min_value = 0,
                       max_value = 15,
                       value = 10,
                       key = "OVRVW_WOC")
        woc_dict = {sku: woc for sku in list(sku_data.keys())}
        
        @st.fragment
        def run_init_shipments(weeks_cover):
            st.session_state.shipments = pd.DataFrame()
            st.session_state.predictions = pd.DataFrame()
            st.session_state.logic = ""
            shipment_memory = pd.DataFrame()
            
            pred_dict = run_forecast(data = SC_demand_filled, cross_validation = cv)
                
            st.session_state.shipments, st.session_state.predictions, st.session_state.logic = generate_recommended_shipments(pred_dict, woc_dict)

            
        if st.button(label = "Run Shipment Recommendations",
                     key = "GLOBAL_SHIPMENT_RECO"):
            run_init_shipments(weeks_cover = woc)

        if st.session_state.shipments.empty:
                run_init_shipments(weeks_cover = woc)

        # displays
        
        st.table(data = st.session_state.shipments.sort_values(by = "Creation Date"))

        with st.expander("Shipment Logic"):
            selected_sku = st.selectbox(
                label="Select SKU to Inspect",
                options=list(sku_data.keys()),
                key="Logic_Selector"
                    )

            if selected_sku:
                sku_logs = st.session_state.logic.get(selected_sku)
                
                if sku_logs:
                    st.markdown(f"### Logic for SKU: `{selected_sku}`")
                    for log_entry in sku_logs:
                        st.markdown(f"- {log_entry}")
                else:
                    st.info("No logs for this SKU.")

        
        @st.fragment
        def download_data():
                if st.button("Download Shipments"):
                        st.table(st.session_state.shipments)
                        st.download_button(
                            label = "Download Shipments Data",
                            data = st.session_state.shipments.to_csv(index=False),
                            file_name = "shipments.csv",
                            mime="text/csv",
                            key="Download_Shipments"
                            )
        download_data()
    

    with shp_inspct_tab:

        insp_sku = st.selectbox(label = "Select SKU to Inspect",
                           options = list(sku_data.keys()),
                           key = "Inspect_Selector")

        
        product_tab(sku = insp_sku, 
                    pred_df = st.session_state.predictions[insp_sku], 
                    case_qty =  sku_data[insp_sku][1], 
                    pallet_qty =  sku_data[insp_sku][2])
            
