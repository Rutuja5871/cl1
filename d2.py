import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Define the base URL for OpenWeatherMap Historical API
BASE_URL = "https://history.openweathermap.org/data/2.5/history/city"
API_KEY = "6c834e240e6164bd74c5be4f3864b87a"  # Replace with your actual API key
# CITY_ID = "1259229"  # Replace with the city ID for the desired location (e.g., for city with ID 2885679)

# Set the start and end timestamps (example for a one-week period)
start_date = int(datetime(2024, 10, 27).timestamp())
end_date = int(datetime(2024, 11, 3).timestamp())
print(start_date, end_date)

# Function to fetch historical weather data
def fetch_historical_data(start, end):
    params = {
        "q": "Pune,In",
        "type": "hour",
        "start": start,
        "end": end,
        "appid": API_KEY,
        'cnt':1,
        "units":"metric"
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

# Fetch data
data = fetch_historical_data(start_date, end_date)
print(data)

# Step 2: Convert JSON Data to DataFrame
# Parse the weather data into a structured DataFrame
if data and 'list' in data:
    weather_records = []
    for record in data['list']:
        weather_info = {
            "timestamp": datetime.utcfromtimestamp(record['dt']),
            "temperature": record['main']['temp'],
            "humidity": record['main']['humidity'],
            "pressure": record['main']['pressure'],
            "wind_speed": record['wind']['speed']
        }
        weather_records.append(weather_info)
    # print(weather_records)

    # Create DataFrame
    weather_df = pd.DataFrame(weather_records)
    print(weather_df.head())
else:
    print("No data found.")

# Step 3: Data Cleaning and Transformation
# Check for missing values
print("\nMissing Values:\n", weather_df.isnull().sum())

# Convert timestamp to datetime format if not already done
weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
#weather_df['temp'] = weather_df['temperature'] - 273.15
weather_df['temp'] = weather_df['temperature']
weather_df.set_index('timestamp', inplace=True)

print(weather_df.head())
print(weather_df.tail())

# Step 4: Data Analysis and Visualization
# Plot temperature trend
plt.figure(figsize=(14, 6))
sns.lineplot(data=weather_df, x=weather_df.index, y="temp", label="Temperature (°C)")
plt.title("Temperature Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()

# Plot humidity trend
plt.figure(figsize=(14, 6))
sns.lineplot(data=weather_df, x=weather_df.index, y="humidity", color="orange", label="Humidity (%)")
plt.title("Humidity Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Humidity (%)")
plt.legend()
plt.show()

# Plot pressure trend
plt.figure(figsize=(14, 6))
sns.lineplot(data=weather_df, x=weather_df.index, y="pressure", color="green", label="Pressure (hPa)")
plt.title("Pressure Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Pressure (hPa)")
plt.legend()
plt.show()

# Plot wind speed trend
plt.figure(figsize=(14, 6))
sns.lineplot(data=weather_df, x=weather_df.index, y="wind_speed", color="blue", label="Wind Speed (m/s)")
plt.title("Wind Speed Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Wind Speed (m/s)")
plt.legend()
plt.show()
