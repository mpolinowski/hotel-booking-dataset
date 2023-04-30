# Hotel Booking Demand Dataset

<!-- TOC -->

- [Hotel Booking Demand Dataset](#hotel-booking-demand-dataset)
  - [Dataset](#dataset)
    - [Size](#size)
    - [Missing Data](#missing-data)
    - [Exploration](#exploration)
      - [Data Columns](#data-columns)
      - [Top Countries](#top-countries)
      - [Average Daily Rates](#average-daily-rates)
      - [Average Stays](#average-stays)
      - [Average Cost per Stay](#average-cost-per-stay)
      - [Percentage of Returning Guest](#percentage-of-returning-guest)
      - [Correlate Bookings to Day of the Week](#correlate-bookings-to-day-of-the-week)
      - [Bookings within a Date Range](#bookings-within-a-date-range)

<!-- /TOC -->



> see [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

```python
import numpy as np
import pandas as pd
import datetime
```

## Dataset

```python
hotel_bookings = pd.read_csv('datasets/hotel_bookings.csv')
hotel_bookings.head(5)
```

### Size

```python
# complete number of rows
print(hotel_bookings.index)
# RangeIndex(start=0, stop=119390, step=1)

# complete number of columns
print(len(hotel_bookings.columns))
# 32
```

### Missing Data

```python
# only show rows that have missing values
hotel_bookings_nan = hotel_bookings[hotel_bookings.isna().any(axis=1)]
hotel_bookings_nan
# 119173 rows × 32 columns
# only 119390 - 119173 =  217 rows don't have missing entries
```

| | hotel | is_canceled | lead_time | arrival_date_year | arrival_date_month | arrival_date_week_number | arrival_date_day_of_month | stays_in_weekend_nights | stays_in_week_nights | adults | ... | deposit_type | agent | company | days_in_waiting_list | customer_type | adr | required_car_parking_spaces | total_of_special_requests | reservation_status | reservation_status_date |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | Resort Hotel | 0 | 342 | 2015 | July | 27 | 1 | 0 | 0 | 2 | ... | No Deposit | NaN | NaN | 0 | Transient | 0.00 | 0 | 0 | Check-Out | 01-07-15 |
| 1 | Resort Hotel | 0 | 737 | 2015 | July | 27 | 1 | 0 | 0 | 2 | ... | No Deposit | NaN | NaN | 0 | Transient | 0.00 | 0 | 0 | Check-Out | 01-07-15 |
| 2 | Resort Hotel | 0 | 7 | 2015 | July | 27 | 1 | 0 | 1 | 1 | ... | No Deposit | NaN | NaN | 0 | Transient | 75.00 | 0 | 0 | Check-Out | 02-07-15 |
| 3 | Resort Hotel | 0 | 13 | 2015 | July | 27 | 1 | 0 | 1 | 1 | ... | No Deposit | 304.0 | NaN | 0 | Transient | 75.00 | 0 | 0 | Check-Out | 02-07-15 |
| 4 | Resort Hotel | 0 | 14 | 2015 | July | 27 | 1 | 0 | 2 | 2 | ... | No Deposit | 240.0 | NaN | 0 | Transient | 98.00 | 0 | 1 | Check-Out | 03-07-15 |
| ... |
| 119385 | City Hotel | 0 | 23 | 2017 | August | 35 | 30 | 2 | 5 | 2 | ... | No Deposit | 394.0 | NaN | 0 | Transient | 96.14 | 0 | 0 | Check-Out | 06-09-17 |
| 119386 | City Hotel | 0 | 102 | 2017 | August | 35 | 31 | 2 | 5 | 3 | ... | No Deposit | 9.0 | NaN | 0 | Transient | 225.43 | 0 | 2 | Check-Out | 07-09-17 |
| 119387 | City Hotel | 0 | 34 | 2017 | August | 35 | 31 | 2 | 5 | 2 | ... | No Deposit | 9.0 | NaN | 0 | Transient | 157.71 | 0 | 4 | Check-Out | 07-09-17 |
| 119388 | City Hotel | 0 | 109 | 2017 | August | 35 | 31 | 2 | 5 | 2 | ... | No Deposit | 89.0 | NaN | 0 | Transient | 104.40 | 0 | 0 | Check-Out | 07-09-17 |
| 119389 | City Hotel | 0 | 205 | 2017 | August | 35 | 29 | 2 | 7 | 2 | ... | No Deposit | 9.0 | NaN | 0 | Transient | 151.20 | 0 | 2 | Check-Out | 07-09-17 |

```python
# which columns have the most missing entries
hotel_bookings.isna().sum()

# the columns company, agent and country have the most missing data:
```

|    |    |
| -- | -- |
| hotel | 0 |
| is_canceled | 0 |
| lead_time | 0 |
| arrival_date_year | 0 |
| arrival_date_month | 0 |
| arrival_date_week_number | 0 |
| arrival_date_day_of_month | 0 |
| stays_in_weekend_nights | 0 |
| stays_in_week_nights | 0 |
| adults | 0 |
| children | 4 |
| babies | 0 |
| meal | 0 |
| country | 488 |
| market_segment | 0 |
| distribution_channel | 0 |
| is_repeated_guest | 0 |
| previous_cancellations | 0 |
| previous_bookings_not_canceled | 0 |
| reserved_room_type | 0 |
| assigned_room_type | 0 |
| booking_changes | 0 |
| deposit_type | 0 |
| agent | 16340 |
| company | 112593 |
| days_in_waiting_list | 0 |
| customer_type | 0 |
| adr | 0 |
| required_car_parking_spaces | 0 |
| total_of_special_requests | 0 |
| reservation_status | 0 |
| reservation_status_date | 0 |
_dtype: int64_

```python
# drop columns with missing data
hotel_bookings_dropped_nan = hotel_bookings.drop(['company', 'agent'], axis=1)
hotel_bookings_dropped_nan.head(2)
```

```python
hotel_bookings_dropped_nan[hotel_bookings_dropped_nan.isna().any(axis=1)]
# 4 rows × 29 columns
# only the 4 rows with missing data in the children column and 488 country column remain
```

### Exploration

#### Data Columns

```python
# what columns do we have
pd.Series(hotel_bookings.columns)
```

|  |  |
| -- | -- |
| 0 | hotel |
| 1 | is_canceled |
| 2 | lead_time |
| 3 | arrival_date_year |
| 4 | arrival_date_month |
| 5 | arrival_date_week_number |
| 6 | arrival_date_day_of_month |
| 7 | stays_in_weekend_nights |
| 8 | stays_in_week_nights |
| 9 | adults |
| 10 | children |
| 11 | babies |
| 12 | meal |
| 13 | country |
| 14 | market_segment |
| 15 | distribution_channel |
| 16 | is_repeated_guest |
| 17 | previous_cancellations |
| 18 | previous_bookings_not_canceled |
| 19 | reserved_room_type |
| 20 | assigned_room_type |
| 21 | booking_changes |
| 22 | deposit_type |
| 23 | agent |
| 24 | company |
| 25 | days_in_waiting_list |
| 26 | customer_type |
| 27 | adr |
| 28 | required_car_parking_spaces |
| 29 | total_of_special_requests |
| 30 | reservation_status |
| 31 | reservation_status_date |
_dtype: object_


#### Top Countries

```python
# top5 country codes
hotel_bookings_dropped_nan['country'].value_counts().head(5)
```

|  |  |
| -- | -- |
| PRT | 48590 |
| GBR | 12129 |
| FRA | 10415 |
| ESP | 8568 |
| DEU | 7287 |
_Name: country, dtype: int64_

```python
hotel_bookings_dropped_nan['country'].value_counts().head(15).plot.bar(figsize=(12,4),rot=65)
```

![Hotel Booking Demand Dataset](assets/hotel_booking_demand_01.png)


#### Average Daily Rates

```python
plot = hotel_bookings_dropped_nan.plot.scatter(
    figsize=(12,8),
    x='adr',
    y='hotel')

# there are only 2 hotels and all adr's are within 0-500$ with one outlier above 5000$
```

![Hotel Booking Demand Dataset](assets/hotel_booking_demand_02.png)

```python
# find outlier
hotel_bookings_dropped_nan.sort_values('adr', ascending=False).iloc[0]
```

|    |    |
| -- | -- |
| hotel | City Hotel |
| is_canceled | 1 |
| lead_time | 35 |
| arrival_date_year | 2016 |
| arrival_date_month | March |
| arrival_date_week_number | 13 |
| arrival_date_day_of_month | 25 |
| stays_in_weekend_nights | 0 |
| stays_in_week_nights | 1 |
| adults | 2 |
| children | 0.0 |
| babies | 0 |
| meal | BB |
| country | PRT |
| market_segment | Offline TA/TO |
| distribution_channel | TA/TO |
| is_repeated_guest | 0 |
| previous_cancellations | 0 |
| previous_bookings_not_canceled | 0 |
| reserved_room_type | A |
| assigned_room_type | A |
| booking_changes | 1 |
| deposit_type | Non Refund |
| days_in_waiting_list | 0 |
| customer_type | Transient |
| adr | 5400.0 |
| required_car_parking_spaces | 0 |
| total_of_special_requests | 0 |
| reservation_status | Canceled |
| reservation_status_date | 19-02-16 |
_Name: 48515, dtype: object_

```python
plot = hotel_bookings_dropped_nan.plot.hist(
    column=["adr"],
    by="hotel",
    bins=100,
    figsize=(10, 8)
)

# the outlier squeezes the first histogram and makes it hard to compare them
```

![Hotel Booking Demand Dataset](assets/hotel_booking_demand_03.png)

```python
# let's find the outlier iloc and drop the row
hotel_bookings_dropped_nan['adr'].idxmax()
# 48515
```

```python
hotel_bookings_dropped_outlier = hotel_bookings_dropped_nan.drop(48515, axis=0)

plot = hotel_bookings_dropped_outlier.plot.hist(
    column=["adr"],
    by="hotel",
    bins=100,
    figsize=(10, 8)
)

# nice :)
```

![Hotel Booking Demand Dataset](assets/hotel_booking_demand_04.png)

```python
# calculate the average daily rate `adr` for a guest staying at each hotel
adr_by_hotel = hotel_bookings_dropped_nan.groupby('hotel').mean(numeric_only=True)['adr']
adr_by_hotel
```

__Average Daily Rate__

|    |    |
| -- | -- |
| hotel |  |
| City Hotel | 105.304465 |
| Resort Hotel | 94.952930 |
_Name: adr, dtype: float64_


#### Average Stays

```python
# how long do guest stay on average
hotel_bookings_dropped_nan['total_days'] = hotel_bookings_dropped_nan['stays_in_weekend_nights'] + hotel_bookings_dropped_nan['stays_in_week_nights']
hotel_bookings_dropped_nan['total_days'].head(5)
```

|    |    |
| -- | -- |
| 0 | 0 |
| 1 | 0 |
| 2 | 1 |
| 3 | 1 |
| 4 | 2 |
_Name: total\_days, dtype: int64_

```python
average_stays = hotel_bookings_dropped_nan.groupby('hotel').mean(numeric_only=True).round(1)['total_days']
average_stays

# the average staying time is 3 and 4.3 days, respectively
```

|  hotel  |    |
| -- | -- |
| City Hotel | 3.0 |
| Resort Hotel | 4.3 |
_Name: total|_days, dtype: float64_


#### Average Cost per Stay

```python
# given the # of days and average daily adr we can calculate the average total cost per stay
hotel_bookings_dropped_nan['total_cost'] = hotel_bookings_dropped_nan['total_days'] * hotel_bookings_dropped_nan['adr']
hotel_bookings_dropped_nan['total_cost'].head(5)
```

|    |    |
| -- | -- |
| 0 | 0.0 |
| 1 | 0.0 |
| 2 | 75.0 |
| 3 | 75.0 |
| 4 | 196.0 |
_Name: total\_days, dtype: float64_

```python
average_total_cost = hotel_bookings_dropped_nan.groupby('hotel').mean(numeric_only=True).round(2)['total_cost']
average_total_cost
```

| hotel   |    |
| -- | -- |
| City Hotel | 318.66 |
| Resort Hotel | 435.45 |
_Name: total_days, dtype: float64_


#### Percentage of Returning Guest

```python
# total number of bookings per hotel
hotel_bookings_dropped_nan.value_counts('hotel')
```

| hotel   |    |
| -- | -- |
| City Hotel | 79330 |
| Resort Hotel | 40060 |
_dtype: int64_

```python
# select only city hotel
city_hotel_bookings = hotel_bookings_dropped_nan[hotel_bookings_dropped_nan['hotel'] == 'City Hotel']
city_hotel_bookings['hotel'].head(5)
```

|    |    |
| -- | -- |
| 40060 | City Hotel |
| 40061 | City Hotel |
| 40062 | City Hotel |
| 40063 | City Hotel |
| 40064 | City Hotel |
_Name: hotel, dtype: object_

```python
# select only resort hotel
resort_hotel_bookings = hotel_bookings_dropped_nan[hotel_bookings_dropped_nan['hotel'] == 'Resort Hotel']
resort_hotel_bookings['hotel'].head(5)
```

|    |    |
| -- | -- |
| 0 | Resort Hotel |
| 1 | Resort Hotel |
| 2 | Resort Hotel |
| 3 | Resort Hotel |
| 4 | Resort Hotel |
_Name: hotel, dtype: object_

```python
returning_customer_city_hotel = sum(city_hotel_bookings['is_repeated_guest'] == 1)
returning_customer_city_hotel
# 2032
```

```python
total_customer_city_hotel = hotel_bookings_dropped_nan.value_counts('hotel')['City Hotel']
total_customer_city_hotel
# 79330
```

```python
percentage_returning_customer_city_hotel = (
    returning_customer_city_hotel * 100 / total_customer_city_hotel
)
percentage_returning_customer_city_hotel.round(2)
# 2.56%
```

```python
returning_customer_resort_hotel = sum(resort_hotel_bookings['is_repeated_guest'] == 1)
total_customer_resort_hotel = hotel_bookings_dropped_nan.value_counts('hotel')['Resort Hotel']

percentage_returning_customer_resort_hotel = (
    returning_customer_resort_hotel * 100 / total_customer_resort_hotel
)
percentage_returning_customer_resort_hotel.round(2)
# 4.44%
```

```python
# visualize
hotel_index = ['City Hotel', 'Resort Hotel']
booking_columns = ['Total Bookings', 'Returning Customer', 'Percentage']
data_array = [
    (
        total_customer_city_hotel,
        returning_customer_city_hotel,
        percentage_returning_customer_city_hotel.round(2)
    ),
    (
        total_customer_resort_hotel,
        returning_customer_resort_hotel,
        percentage_returning_customer_resort_hotel.round(2)
    )
]

return_customer_df = pd.DataFrame(data_array, hotel_index, booking_columns)
return_customer_df
```

| Hotel   |  Total Bookings  |  Returning Customer  | Percentage|
| -- | -- | -- | -- |
| City Hotel | 79330 | 2032 | 2.56 |
| Resort Hotel | 40060 | 1778 | 5.07 |

```python
plot = return_customer_df[
    ['Total Bookings', 'Returning Customer']
].plot.bar(figsize=(12,8), rot=0)
```

![Hotel Booking Demand Dataset](assets/hotel_booking_demand_05.png)

```python
arrival_date_year 	2016
arrival_date_month 	March
arrival_date_day_of_month 	25
Aug 27, 1989
pd.to_datetime(date_series)
```

```python
city_hotel_bookings[[
    'arrival_date_year',
    'arrival_date_month',
    'arrival_date_day_of_month'
]].head(5)
```

| | arrival_date_year | arrival_date_month | arrival_date_day_of_month |
| -- | -- | -- | -- |
| 40060 | 2015 | July | 1 |
| 40061 | 2015 | July | 1 |
| 40062 | 2015 | July | 1 |
| 40063 | 2015 | July | 1 |
| 40064 | 2015 | July | 2 |


#### Correlate Bookings to Day of the Week

```python
city_hotel_bookings['datetime'] = (
    pd.to_datetime(
        city_hotel_bookings['arrival_date_month'] + ' ' + city_hotel_bookings['arrival_date_day_of_month'].astype(str) + ' , ' + city_hotel_bookings['arrival_date_year'].astype(str)
    )
)

city_hotel_bookings['datetime']
```

|  |  |
| -- | -- |
| 40060 | 2015-07-01 |
| 40061 | 2015-07-01 |
| 40062 | 2015-07-01 |
| 40063 | 2015-07-01 |
| 40064 | 2015-07-02 |
| ... |
| 119385 | 2017-08-30 |
| 119386 | 2017-08-31 |
| 119387 | 2017-08-31 |
| 119388 | 2017-08-31 |
| 119389 | 2017-08-29 |
_Name: datetime, Length: 79330, dtype: datetime64[ns]_

```python
# get weekday out of datetime object
city_hotel_bookings['datetime'].loc[40064].weekday()
# 3 == Thursday
```

```python
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def weekday(day_of_the_week):
        return days[day_of_the_week]
```

```python
city_hotel_bookings['day_of_the_week'] = np.vectorize(weekday)(
        city_hotel_bookings['datetime'].dt.weekday
)

city_hotel_bookings['day_of_the_week'].tail(5)
```

|  |  |
| -- | -- |
| 119385 | Wednesday |
| 119386 | Thursday |
| 119387 | Thursday |
| 119388 | Thursday |
| 119389 | Tuesday |
_Name: day\_of\_the\_week, dtype: object_

```python
bookings_by_weekday_city_hotel = city_hotel_bookings.value_counts('day_of_the_week')
bookings_by_weekday_city_hotel
```

| day_of_the_week |  |
| -- | -- |
| Friday | 13955 |
| Thursday | 13009 |
| Monday | 11823 |
| Wednesday | 11229 |
| Saturday | 10993 |
| Sunday | 9194 |
| Tuesday | 9127 |
_dtype: int64_

```python
bookings_by_weekday_city_hotel.plot.bar(figsize=(12,8), rot=0)
```

![Hotel Booking Demand Dataset](assets/hotel_booking_demand_06.png)

```python
resort_hotel_bookings['datetime'] = (
    pd.to_datetime(
        resort_hotel_bookings['arrival_date_month'] + ' ' + resort_hotel_bookings['arrival_date_day_of_month'].astype(str) + ' , ' + resort_hotel_bookings['arrival_date_year'].astype(str)
    )
)

resort_hotel_bookings['day_of_the_week'] = np.vectorize(weekday)(
        resort_hotel_bookings['datetime'].dt.weekday
)

bookings_by_weekday_resort_hotel = resort_hotel_bookings.value_counts('day_of_the_week')
bookings_by_weekday_resort_hotel
```

| day_of_the_week |  |
| -- | -- |
| Saturday | 7062 |
| Monday | 6348 |
| Thursday | 6245 |
| Friday | 5676 |
| Sunday | 4947 |
| Wednesday | 4910 |
| Tuesday | 4872 |
_dtype: int64_

```python
bookings_by_weekday_resort_hotel.plot.bar(figsize=(12,8), rot=0)
```

![Hotel Booking Demand Dataset](assets/hotel_booking_demand_07.png)


#### Bookings within a Date Range

```python
first_15 = hotel_bookings_dropped_nan['arrival_date_day_of_month'].apply(lambda day: day in range(1,16)).sum()
first_15
# 58152
```

```python
last_15 = hotel_bookings_dropped_nan['arrival_date_day_of_month'].apply(lambda day: day in range(15,32)).sum()
last_15
# 65434
```


