library(tidyverse)
library(data.table)
library(lubridate)
library(timetk)
library(skimr)
library(highcharter)
library(tidymodels)
library(modeltime)
library(zoo)

raw <- fread('AirPassengers.csv') 

colnames(raw) <- c('Date','Count')

raw$Date <- as.Date(as.yearmon(raw$Date))

raw %>% glimpse()

raw %>% 
  plot_time_series(
    Date, Count, 
    # .color_var = lubridate::year(Date),
    # .color_lab = "Year",
    .interactive = T,
    .plotly_slider = T)

# There is trend and seasonality in our dataset.

train <- raw %>% filter(Date < "1958-01-01")
test <- raw %>% filter(Date >= "1958-01-01")

# Model 1: arima_boost ----
model_fit_arima_boosted <- arima_boost(
  min_n = 2,
  learn_rate = 0.015
) %>%
  set_engine(engine = "auto_arima_xgboost") %>%
  fit(Count ~ Date + as.numeric(Date) + factor(lubridate::month(Date, label = TRUE), ordered = F),
      data = train)

# Model 2: ets ----
model_fit_ets <- exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(Count ~ Date, data = train)

# Model 3: prophet ----
model_fit_prophet <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(Count ~ Date, data = train)


# calibration
calibration <- modeltime_table(
  model_fit_arima_boosted,
  model_fit_ets,
  model_fit_prophet) %>%
  modeltime_calibrate(test)

# Predict ----
calibration %>% 
  modeltime_forecast(actual_data = raw) %>%
  plot_modeltime_forecast(.interactive = T,
                          .plotly_slider = T)

# Accuracy ----
calibration %>% modeltime_accuracy() %>% 
  table_modeltime_accuracy(.interactive = F) # ARIMA has the Lowest RMSE.


# Forecast Forward ----
calibration %>%
  filter(.model_id %in% 1) %>% # best model
  modeltime_refit(raw) %>%
  modeltime_forecast(h = "2 year", 
                     actual_data = raw) %>%
  plot_modeltime_forecast(.interactive = T,
                          .plotly_slider = T,
                          .legend_show = F)
