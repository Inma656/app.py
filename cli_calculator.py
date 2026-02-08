# cli_calculator.py
# Drone Delivery Logistics Calculation Program
# This script estimates distances, times, and number of drones needed
# for a given delivery scenario using a square root TSP approximation.
# Based on Listing 1 from the provided document.

import math

def get_numeric_input(message, min_val=0):
    """
    Function to ensure valid numeric input from the user.
    It keeps asking until a valid float > min_val is provided.
    """
    while True:
        try:
            value = float(input(message))
            if value < min_val:
                print(f"Please enter a value greater than or equal to {min_val}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def main():
    # --- Collect drone information from the user ---
    print("Enter drone information:")
    battery_capacity = get_numeric_input("Battery capacity (mAh or Wh): ")
    flight_time = get_numeric_input("Flight time with full battery (minutes): ")
    charge_time = get_numeric_input("Battery charge time (minutes): ")
    load_capacity = get_numeric_input("Load capacity (kg): ")
    speed_no_load = get_numeric_input("Speed without load (mph): ")
    speed_with_load = get_numeric_input("Speed with maximum load (mph): ")
    drone_cost = get_numeric_input("Drone cost (USD): ")

    # Basic sanity checks on time and speed
    if charge_time <= 0 or flight_time <= 0 or speed_with_load <= 0:
        print("Error: Ensure charge time, flight time, and loaded speed are greater than zero.")
        return

    # --- Collect delivery information (demand side) ---
    print("\nDelivery information:")
    num_drops = get_numeric_input("Number of drops to make: ", 1)
    city_area = get_numeric_input("City area (square miles): ", 0.1)

    # --- Calculations ---
    
    # 1. Calculate total distance using the square root formula
    # (Beardwood-Halton-Hammersley-type Euclidean TSP approximation)
    # L approx 0.72 * sqrt(n * A)
    total_distance = 0.72 * math.sqrt(num_drops * city_area)

    # 2. Calculate average distance per drop
    distance_per_drop = total_distance / num_drops
    print(f"\nEstimated distance between each drop: {distance_per_drop:.2f} miles")

    # 3. Calculate round-trip flight time per drop (go to the house and return)
    # Using speed_with_load as a conservative estimate
    round_trip_flight_time = (distance_per_drop * 2) / speed_with_load * 60

    # 4. Additional time for handling (acceleration, delivery, deceleration)
    # 4 min down + 2 min delivery + 4 min up = 10 mins approx
    delivery_time = (4/60 + 2/60 + 4/60) * 60 # converted to minutes for consistency
    # (The PDF uses hours in one part and minutes in another, here we standardize to minutes)
    
    # Total time per delivery (flight + handling) in minutes
    total_time_per_delivery = round_trip_flight_time + delivery_time

    # 5. Deliveries per charge
    deliveries_per_charge = int(flight_time / total_time_per_delivery)

    if deliveries_per_charge <= 0:
        print("Error: The battery is not sufficient to complete a single trip. Check the input values.")
        return

    # 6. Calculate how many locations can be covered before returning (Dynamic payload logic)
    current_load = load_capacity
    locations_covered = 0
    remaining_flight_time = flight_time

    # Simulation loop for a single trip
    while remaining_flight_time > 0 and current_load >= 0.5:
        # Use loaded speed for the first trip, then unloaded speed afterward (simplified model)
        current_speed = speed_with_load if current_load == load_capacity else speed_no_load
        
        trip_time = ((distance_per_drop * 2) / current_speed) * 60
        trip_total_time = trip_time + delivery_time
        
        if remaining_flight_time >= trip_total_time:
            remaining_flight_time -= trip_total_time
            current_load -= 0.5  # assume each drop weighs 0.5 kg
            locations_covered += 1
        else:
            break
    
    print(f"\nLocations covered before returning (payload simulation): {locations_covered}")

    # 7. Calculate deliveries per workday (8-hour working day)
    workday_time = 8 * 60  # minutes in an 8-hour day
    
    # Cycle time = time flying for X deliveries + time charging
    cycle_total_time = (deliveries_per_charge * total_time_per_delivery) + charge_time
    
    if cycle_total_time > 0:
        cycles_per_day = workday_time / cycle_total_time
    else:
        cycles_per_day = 0

    # Total deliveries per day for a single drone
    deliveries_per_day = int(cycles_per_day * deliveries_per_charge)

    if deliveries_per_day <= 0:
        print("Error: Charging time is too long to complete any drops within an 8-hour workday.")
        return

    # 8. Calculate required number of drones
    required_drones = math.ceil(num_drops / deliveries_per_day)

    # 9. Monthly drone distribution (assuming 5 workdays/week, 4 weeks/month)
    days_per_week = 5
    weeks_per_month = 4
    workdays_per_month = days_per_week * weeks_per_month

    trips_per_day = deliveries_per_day
    trips_per_week = trips_per_day * days_per_week
    trips_per_month = trips_per_week * weeks_per_month

    print(f"\nMonthly drone distribution:")
    print(f"Drones needed per day to finish all drops: {required_drones}")
    # Note: The PDF logic for 'Drones per week' seemed to multiply drones * days, 
    # which might be 'drone-days', but here we stick to the output format requested.
    
    print(f"\nTrips per day (per drone): {trips_per_day}")
    print(f"Trips per week (per drone): {trips_per_week}")
    print(f"Trips per month (per drone): {trips_per_month}")

    # --- Final Summary ---
    print("\nFinal Summary:")
    print(f"Required number of drones: {int(required_drones)}")
    print(f"Total number of drops requested: {int(num_drops)}")
    
    # Estimated time for the whole fleet (just a reference metric)
    total_hours_all_drops = (num_drops * total_time_per_delivery) / 60
    print(f"Estimated cumulative drone-hours to complete: {total_hours_all_drops:.2f} hours")
    
    print(f"Trips per full charge: {int(deliveries_per_charge)}")
    print(f"Drops per day per drone: {int(deliveries_per_day)}")

if __name__ == "__main__":
    main()
