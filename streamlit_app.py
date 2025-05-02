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
            "40-05-PEA-50BIB-CS": ["2023-01-01", 3, 60]}
sku_replacement = {"40-05-WTO-A220-CS-stickerless":"40-05-WTO-A220-CS",
                   "40-05-EVO-0750-CS-stickerless": "40-05-EVO-0750-CS",
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

    def run_forecast(sku, data, cross_validation):
        
            sc_sku = sku
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
        
            prediction_df["forc_unit_csum"] = prediction_df["Forecasted Units Sold"].cumsum()
            prediction_df["Shipments"] = 0  
    
            return np.clip(prediction_df, 0, None)
    
    
    
    def shipment_reco(predicted_demand_df, initial_inventory, weeks_of_cover, case_qty, pallet_qty, sku, case_pallet_optim = "Pallet", verbose = False):
    
            sc_sku = sku
            recommended_shipments = []
            pred_df = predicted_demand_df.copy()  # Work on a copy to avoid modifying the original DataFrame
            pred_df["Forecasted Inventory"] = 0  # Reset inventory for recalculating
            current_inventory = initial_inventory
            
            expl_str = ""
            for i in range(len(pred_df)):
                    
                # If we've already created a shipment (no need to calculate forecasted inventory)
                if recommended_shipments:
                    current_inventory = pred_df.iloc[i]["Forecasted Inventory"]
                # If no shipments have been made and we need to initialize forecasted inventory
                else:
                    current_inventory -= pred_df.iloc[i]["Forecasted Units Sold"]
                    current_inventory = max(0, current_inventory)
                    pred_df.loc[pred_df.index[i], "Forecasted Inventory"] = current_inventory

                if ((pred_df.loc[pred_df.index[i-1], "Forecasted Inventory"] == 0) & ( pred_df.loc[pred_df.index[i], "Forecasted Inventory"] == 0)):
            
                    if not expl_str =="":
                        expl_str += f"""
            -- {pred_df.index[i]}- OOS. Waiting for last shipment to arrive before continuing... 
        """
                    continue
                
                if current_inventory == 0:
                    zero_date = pred_df.index[i]  # Save date when inventory = 0
                    shp_avail_dt = zero_date - pd.Timedelta(weeks=weeks_of_cover)  # Adjust for Desired Weeks of Cover
                    
                            # Find optimal shipment quantity
                    optim_qty = EOQ_func(date=zero_date, demand=pred_df["Forecasted Units Sold"][i:i+4].sum(), sku = sc_sku, min_qty=4*6)
                    if case_pallet_optim == "Pallet":
                            optim_qty = max(1, round(optim_qty / case_qty / pallet_qty)) * pallet_qty * case_qty # OPTIMIZE FOR PALLET. IF ROUNDS TO 0, DEFAULTS TO 1.
                    elif case_pallet_optim == "Case":
                            optim_qty = round(optim_qty/case_qty)*case_qty # OPTIMIZE FOR CASE
 
                    
                    # Estimate processing lead time
                    prcss_wks = round(OLS_prcssng_tm_bckwrd(quantity=optim_qty, close_date=shp_avail_dt) / 7)
                    creation_date = shp_avail_dt - pd.Timedelta(weeks=prcss_wks)
                    print(optim_qty, creation_date)
                            # Send shipment out next monday if recommended creation date is before today
                    if creation_date < pd.Timestamp.today():
                        creation_date = pd.to_datetime(np.busday_offset(np.datetime64('today', 'D') , offsets = 0, roll="forward", weekmask='Sat'))
                        prcss_wks =  round(OLS_prcssng_tm(quantity=optim_qty, creation_date=creation_date) / 7)
                        shp_avail_dt = creation_date + pd.Timedelta(weeks = prcss_wks)
            
                        expl_str += f"""
            - We are projected to run out of inventory on {zero_date}. 
            {weeks_of_cover} weeks of cover not possible.
            Earliest possible shipment available date: {shp_avail_dt}.
            Add {prcss_wks} weeks of estimated processing time.
            Shipment creation date: {creation_date}.
            """
                    else:
                        expl_str += f"""
            - We are projected to run out of inventory on {zero_date}. 
            Subtract {weeks_of_cover} weeks of cover: {shp_avail_dt}.
            Add {prcss_wks} weeks of estimated processing time.
            Shipment creation date: {creation_date}.
            """
                        
                    
                            # Append shipment recommendation
                    recommended_shipments.append({"Creation Date": creation_date, "Units": optim_qty, "Cases": round(optim_qty/case_qty), "Pallets": round(optim_qty/case_qty/pallet_qty,2), "Availability Date":shp_avail_dt})
                    
                    if shp_avail_dt in pred_df.index:
                        future_index = pred_df.index.get_loc(shp_avail_dt)
                                
                        # Add shipment quantity to inventory
                        current_inventory_shp = optim_qty + pred_df.loc[pred_df.index[future_index], "Forecasted Inventory"]
                        pred_df.loc[pred_df.index[future_index], "Forecasted Inventory"] = current_inventory_shp
                    
                        # Recalculate running inventory
                        for j in range(future_index+1, len(pred_df)):
                            current_inventory_shp -= pred_df.iloc[j]["Forecasted Units Sold"]
                            current_inventory_shp = max(0, current_inventory_shp)
                            pred_df.loc[pred_df.index[j], "Forecasted Inventory"] = current_inventory_shp
                    
                    if shp_avail_dt not in pred_df.index:
                        break
            if verbose == True:
                with st.expander("Shipment Logic"):
                    st.write(expl_str)
            
            return pd.DataFrame(recommended_shipments), pred_df


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

        EOQ = np.sqrt(90*demand*6*1.2/(volume*strg_rt*6))
    
        return max(min_qty, np.round(EOQ))


    def product_tab(sku, pred_df, case_qty, pallet_qty):

        sc_sku = sku
    
        st.header(f"{sc_sku} 20-Week Forecast")
        st.line_chart(
            SC_demand_filled[SC_demand_filled["SKU"] == sc_sku].merge(pred_df[:20], left_index=True, right_index=True, how="outer")[["Units Sold", "Forecasted Units Sold"]]
        )

        #####################################
        # START SHIPMENT PLANNING
        
        @st.fragment
        def shipment_planning_func(predicted_df):
        
            pred_df = predicted_df
            st.header("Shipment Planning")
        
            # User Inputs for Inventory
            curr_inv = st.number_input(label="Current Inventory Level", 
                                               min_value=0, 
                                               key= f"Curr_Inv{sc_sku}",
                                               value = current_inventory_levels[sc_sku])
        
            pred_df["Forecasted Inventory"] = np.clip(curr_inv - pred_df['forc_unit_csum'],0,None)

################################
            # SPLIT TABS INTO AUTO-RECS

            #shipment_memory = pd.DataFrame()
            weeks_cover = st.number_input(label = "Weeks of Cover",
                                          min_value = 0, 
                                          max_value = 15,
                                          value = 10,
                                          key = f"wks_cvr{sc_sku}")
                    
            case_pallet_toggle = st.segmented_control(label = "Optimization Level",
                                                         options = ["Case", "Pallet"],
                                                         default = "Case",
                                                         key = f"Optim_Level{sc_sku}")

            auto_shp_rec_df, auto_pred_df = shipment_reco(predicted_demand_df = pred_df, initial_inventory = curr_inv, weeks_of_cover=weeks_cover, 
                                                          case_qty=case_qty, pallet_qty=pallet_qty, case_pallet_optim = case_pallet_toggle, sku = sc_sku, verbose = True)

            auto_pred_df_long = auto_pred_df.reset_index().melt(id_vars=["Week Ending"], 
                                                                value_vars=["Forecasted Units Sold", "Forecasted Inventory"], 
                                                                var_name="Metric", value_name="Value")
        
                # Create Line Chart
            auto_chart = alt.Chart(auto_pred_df_long).mark_line().encode(
                                x="Week Ending:T",
                                y=alt.Y("Value:Q", title="Units"),
                                color=alt.Color("Metric:N", legend=alt.Legend(title="Legend", symbolSize=50, labelFontSize=12))
                            )
            final_auto_chart = auto_chart.properties(width=900, height=400).interactive()
            st.altair_chart(final_auto_chart, use_container_width=False)

            if st.button(label = "Approve and Update Shipments",
                        key = f"AUTO_SHIPMENT_APPROVAL{sc_sku}"):
                shipment_memory = auto_shp_rec_df
                shipment_memory["SKU"] = sc_sku
                st.session_state.shipments = st.session_state.shipments[st.session_state.shipments["SKU"] != sc_sku]
                st.session_state.shipments = pd.concat([st.session_state.shipments, shipment_memory])
                st.rerun()
                st.success(f"{sc_sku} Shipments saved!")

        shipment_planning_func(predicted_df = pred_df)



    
    with shpt_ovr_tab:

        woc = st.number_input(label = "Weeks of Cover",
                       min_value = 0,
                       max_value = 15,
                       value = 10,
                       key = "OVRVW_WOC")
        
        @st.fragment
        def run_init_shipments(weeks_cover):
            st.session_state.shipments = pd.DataFrame()
            st.session_state.predictions = pd.DataFrame()
            shipment_memory = pd.DataFrame()
            for i, key in enumerate(sku_data.keys()):
            
                pred_df = run_forecast(sku = key, data = SC_demand_filled[SC_demand_filled.index>=list(sku_data.values())[i][0]], cross_validation = cv)
                
                shipment_memory = shipment_reco(predicted_demand_df = pred_df, initial_inventory = current_inventory_levels[key], 
                                          weeks_of_cover = weeks_cover, 
                                          case_qty = list(sku_data.values())[i][1], pallet_qty = list(sku_data.values())[i][2], 
                                          sku = key, 
                              case_pallet_optim = "Pallet",
                              verbose = False)[0]
                
                shipment_memory["SKU"] = key
                st.session_state.shipments = pd.concat([st.session_state.shipments, shipment_memory])

                pred_df["SKU"] = key
                st.session_state.predictions = pd.concat([st.session_state.predictions, pred_df])
            
        if st.button(label = "Run Shipment Recommendations",
                     key = "GLOBAL_SHIPMENT_RECO"):
            run_init_shipments(weeks_cover = woc)

        if st.session_state.shipments.empty:
                run_init_shipments(weeks_cover = woc)



        st.table(data = st.session_state.shipments)

        
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
                    pred_df = st.session_state.predictions[st.session_state.predictions["SKU"]==insp_sku], 
                    case_qty =  sku_data[insp_sku][1], 
                    pallet_qty =  sku_data[insp_sku][2])
            
