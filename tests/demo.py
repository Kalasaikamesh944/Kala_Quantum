import numpy as np
import math
import json
from tqdm import tqdm
from Kala_Quantum.quantum_core import QuantumState, hadamard

def julian_date(year, month, day, hour=0, minute=0, second=0):
    """Calculate the Julian Date."""
    if month <= 2:
        year -= 1
        month += 12

    A = math.floor(year / 100)
    B = 2 - A + math.floor(A / 4)
    JD = math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + B - 1524.5
    JD += (hour + minute / 60 + second / 3600) / 24
    return JD

def sun_position(jd):
    """Calculate the Sun's position (longitude, latitude)."""
    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    g = (357.528 + 0.9856003 * n) % 360
    g = math.radians(g)
    lambda_sun = (L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360
    return lambda_sun, 0  # Latitude is effectively 0 for the Sun

def moon_position(jd):
    """Calculate the Moon's position (longitude, latitude)."""
    n = jd - 2451545.0
    L = (218.316 + 13.176396 * n) % 360
    M = (134.963 + 13.064993 * n) % 360
    F = (93.272 + 13.229350 * n) % 360
    M = math.radians(M)
    F = math.radians(F)
    lambda_moon = (L + 6.289 * math.sin(M)) % 360
    beta_moon = 5.128 * math.sin(F)
    return lambda_moon, beta_moon

def normalize_position(longitude, latitude):
    """Normalize longitude and latitude to [0, 1] range."""
    norm_long = longitude / 360.0
    norm_lat = (latitude + 90) / 180.0
    return norm_long, norm_lat

def train_model(start_year=2024, num_years=9000):
    """Train the quantum model using Sun and Moon positional data."""
    print("Training Quantum Model for 9,000 years using Julian Date format and saving results to JSON...")
    qs = QuantumState(num_qubits=2)

    start_jd = julian_date(start_year, 1, 1)
    end_jd = julian_date(start_year + num_years, 1, 1)

    results = []
    epoch = 0
    current_jd = start_jd
    while current_jd < end_jd:
        for second in tqdm(range(86400), desc=f"Processing Julian Date: {current_jd:.5f}", unit="second"):  # Iterate over every second of the day
            jd_with_seconds = current_jd + second / 86400

            sun_long, sun_lat = sun_position(jd_with_seconds)
            moon_long, moon_lat = moon_position(jd_with_seconds)

            sun_norm = normalize_position(sun_long, sun_lat)
            moon_norm = normalize_position(moon_long, moon_lat)

            # Encode data into quantum gates
            qs.apply_gate(hadamard(), qubit=0)
            qs.apply_cnot(control=0, target=1)

            # Simulate training by measuring and printing normalized positions
            sun_measurement = qs.measure()
            moon_measurement = qs.measure()

            results.append({
                "julian_date": jd_with_seconds,
                "sun": {
                    "longitude": sun_long,
                    "latitude": sun_lat,
                    "normalized": sun_norm,
                    "measurement": sun_measurement
                },
                "moon": {
                    "longitude": moon_long,
                    "latitude": moon_lat,
                    "normalized": moon_norm,
                    "measurement": moon_measurement
                }
            })

        with open("sun_moon_positions.json", "w") as f:
                json.dump(results, f, indent=4)
    
        current_jd += 1  # Increment by one day in Julian Date
        epoch += 1


    print("\nTraining Complete. Results saved to sun_moon_positions.json")

if __name__ == "__main__":
    train_model(start_year=2024, num_years=1)
