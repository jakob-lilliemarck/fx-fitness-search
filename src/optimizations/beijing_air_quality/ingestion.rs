// Valid source columns from Beijing air quality CSV files
// These are columns that can be converted to f32 for preprocessing
// CSV header: No,year,month,day,hour,PM2.5,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,RAIN,wd,WSPM,station
// Excluded: No (u32 identifier), year (not used), station (no f32 conversion)
// Included: wd (WindDirection converts to f32 degrees)
pub const SOURCE_COLUMNS: &[&str] = &[
    "month", "day", "hour", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP",
    "RAIN", "wd", "WSPM",
]; // 15 columns total (indices 0-14)

// CSV file paths for Beijing air quality data
// These are relative to the project root
pub const PATHS: &[&str] = &[
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Aotizhongxin_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Changping_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Dingling_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Dongsi_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Guanyuan_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Gucheng_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Huairou_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Nongzhanguan_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Shunyi_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Tiantan_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Wanliu_20130301-20170228.csv",
    "src/optimizations/beijing_air_quality/data/PRSA_Data_Wanshouxigong_20130301-20170228.csv",
];

pub const WANSHOUXIGONG_PATH: &str = "src/optimizations/beijing_air_quality/data/PRSA_Data_Wanshouxigong_20130301-20170228.csv";
