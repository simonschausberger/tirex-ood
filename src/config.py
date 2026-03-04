# dataset repositories
REPO_CHRONOS = "autogluon/chronos_datasets"
REPO_CHRONOS_EXTRA = "autogluon/chronos_datasets_extra"
REPO_GIFTEVAL_PRETRAIN = "Salesforce/GiftEvalPretrain"
REPO_GIFTEVAL = "Salesforce/GiftEval"

# information about ID and OOD datasets taken from: https://arxiv.org/abs/2505.23719
# chronos ID data
CHRONOS_TRAIN = [
    "wiki_daily_100k", "monash_pedestrian_counts", "wind_farms_hourly", "monash_rideshare", 
    "mexico_city_bikes", "solar",  "solar_1h", "taxi_1h", "ushcn_daily", "weatherbench_hourly", 
    "weatherbench_daily", "weatherbench_weekly",  "wind_farms_daily", 
    "electricity_15min", "monash_electricity_hourly", "monash_electricity_weekly", 
    "monash_kdd_cup_2018", "monash_london_smart_meters", "m4_daily", "m4_hourly", 
    "m4_monthly", "m4_weekly", "taxi_30min", "monash_temperature_rain", 
    "uber_tlc_hourly", "uber_tlc_daily"
]

# chronos ID data extra: some chronos datasets are contained in an extra repository
CHRONOS_TRAIN_EXTRA = [
    "brazilian_cities_temperature", "spanish_energy_and_weather"
]

# chronos OOD data
CHRONOS_ZS_BENCHMARK = [
    "monash_australian_electricity", "ercot", "exchange_rate", "monash_traffic", 
    "nn5", "monash_nn5_weekly", "monash_weather", "monash_covid_deaths", "monash_fred_md", 
    "m4_quarterly", "m4_yearly", "dominick", "m5", "monash_tourism_monthly", 
    "monash_tourism_quarterly", "monash_tourism_yearly", "monash_car_parts", "monash_hospital", 
    "monash_cif_2016", "monash_m1_yearly", "monash_m1_quarterly", "monash_m1_monthly", 
    "monash_m3_monthly", "monash_m3_yearly", "monash_m3_quarterly"
]

# chronos OOD data extra
CHRONOS_ZS_BENCHMARK_EXTRA = [
    "ETTm", "ETTh"
]

# gifteval ID data
GIFTEVAL_TRAIN = [
    "smart", "ideal", "residential_load_power", "residential_pv_power",
    "azure_vm_traces_2017", "borg_cluster_data_2011", "bdg-2_panther", "bdg-2_fox",
    "bdg-2_rat", "bdg-2_bear", "lcl", "sceaux", "borealis", "buildings_900k",
    "largest_2017", "largest_2018", "largest_2019", "largest_2020",
    "largest_2021", "PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMS_BAY", "LOS_LOOP",
    "BEIJING_SUBWAY_30MIN", "SHMETRO", "HZMETRO", "Q-TRAFFIC", "subseasonal",
    "subseasonal_precip", "wind_power", "solar_power", "kaggle_web_traffic_weekly",
    "kdd2022", "godaddy", "favorita_sales", "china_air_quality", "beijing_air_quality",
    "cdc_fluview_ilinet", "cdc_fluview_who_nrevss"
]


# gifteval OOD data: datasets that overlap with training datasets are already removed
GIFTEVAL_ZS_BENCHMARK = [
    "jena_weather", "ett1", "bitbrains_fast_storage", "bitbrains_rnd", "bizitobs_application",
    "bizitobs_l2c", "bizitobs_service", "car_parts_with_missing", "covid_deaths", "electricity/D",
    "ett2", "hierarchical_sales", "hospital", "LOOP_SEATTLE", "m4_quarterly",
    "m4_yearly", "M_DENSE", "restaurant", "saugeenday", "solar", "SZ_TAXI", "us_births"
]