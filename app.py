# app.py
# Streamlit web application for planning drone delivery routes.
# It accepts addresses (pasted or uploaded), geocodes them, solves a TSP,
# and then estimates drone fleet requirements based on the route length.

import streamlit as st
import pandas as pd
import itertools
import math
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic
import sys
import subprocess
import os

# --- Basic page configuration ---
st.set_page_config(page_title="Drone Delivery Planner", layout="wide")
st.title("Drone Delivery System")

# --- CSS Styling ---
st.markdown(
    """
    <style>
    .stApp { background-color: #e6f2ff; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Introduction ---
st.markdown(
    """
    ## Introduction
    Provide delivery addresses via paste or file upload (CSV/TXT).
    **CSV tips:** any delimiter (comma/semicolon/tab) and UTF-8/UTF-8-SIG supported.
    If there is no single 'address' column, select columns to combine. First row is the depot/start.
    """
)

# --- Sidebar: Drone parameters ---
st.sidebar.header("Drone parameters")

battery_capacity = st.sidebar.number_input(
    "Battery capacity (mAh or Wh)", min_value=1.0, value=5000.0
)

flight_time = st.sidebar.number_input(
    "Flight time (minutes) on full battery", min_value=1.0, value=30.0
)

charge_time = st.sidebar.number_input(
    "Charge time (minutes)", min_value=1.0, value=60.0
)

load_capacity = st.sidebar.number_input(
    "Load capacity (kg)", min_value=0.1, value=2.0
)

speed_no_load = st.sidebar.number_input(
    "Speed without load (mph)", min_value=0.1, value=40.0
)

speed_with_load = st.sidebar.number_input(
    "Speed with load (mph)", min_value=0.1, value=30.0
)

drone_cost = st.sidebar.number_input(
    "Drone cost (USD)", min_value=0.0, value=1500.0
)

# --- Helper Functions ---

def _read_csv_flex(uploaded_file) -> pd.DataFrame:
    """
    Robust CSV reader:
    auto-detect delimiter with sep=None (python engine)
    handle UTF-8 and UTF-8-SIG BOM
    do not convert empty strings to NaN (keep_default_na=False)
    """
    uploaded_file.seek(0)
    try:
        return pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
            keep_default_na=False
        )
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(
            uploaded_file,
            sep=",",
            encoding="utf-8-sig",
            keep_default_na=False
        )

def _read_txt_lines(uploaded_file) -> list:
    """Read a TXT file and return a list of non-empty lines."""
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8-sig", errors="ignore")
    else:
        text = str(raw)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines

def _combine_columns(df: pd.DataFrame, cols: list) -> list:
    """
    Combine multiple columns into a single address string,
    skipping empty entries and joining with commas.
    """
    parts = df[cols].astype(str).applymap(lambda x: x.strip()).replace(
        {"": None, "nan": None}
    )
    return parts.apply(
        lambda row: ", ".join([p for p in row.tolist() if p]), axis=1
    ).tolist()

@st.cache_data(show_spinner=False)
def geocode_address(address: str):
    """
    Geocode a single address string to latitude/longitude using Nominatim.
    Caching avoids repeated calls for the same address.
    """
    try:
        geolocator = Nominatim(user_agent="drone_delivery_streamlit_app")
        geocode = RateLimiter(
            geolocator.geocode,
            min_delay_seconds=1,
            max_retries=2,
            error_wait_seconds=2.0
        )
        loc = geocode(address)
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception:
        pass
    return None

def build_distance_matrix(coords):
    """
    Given a list of (lat, lon) coordinates, build a full distance matrix
    using geodesic distance in kilometers.
    """
    n = len(coords)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i][j] = geodesic(coords[i], coords[j]).km
    return m

def exact_tsp(dist_matrix, return_to_start=False):
    """
    Brute-force exact Traveling Salesman Problem solution.
    Only feasible for small n (e.g., n <= 9 or 10).
    """
    n = len(dist_matrix)
    if n <= 1:
        return tuple(range(n)), 0.0
    
    best_route = None
    best_dist = float('inf')
    indices = list(range(1, n))
    
    for perm in itertools.permutations(indices):
        route = [0] + list(perm)
        if return_to_start:
            route.append(0)
            
        d = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        
        if d < best_dist:
            best_dist = d
            best_route = route
            
    return tuple(best_route), best_dist

def nearest_neighbor_tsp(dist_matrix, start=0, return_to_start=False):
    """
    Nearest-neighbor heuristic for the Traveling Salesman Problem.
    Faster for larger n but not guaranteed to find the optimal route.
    """
    n = len(dist_matrix)
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    current = start
    
    while unvisited:
        next_node = min(unvisited, key=lambda x: dist_matrix[current][x])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node
        
    if return_to_start:
        route.append(start)
        
    total = sum(
        dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)
    ) if len(route) > 1 else 0.0
    
    return tuple(route), total

# --- Main Application Logic ---

st.header("Addresses input")
input_method = st.radio(
    "How do you want to provide addresses?",
    ("Paste (one per line)", "Upload CSV/TXT")
)

addresses = []

if input_method == "Paste (one per line)":
    pasted = st.text_area(
        "Paste addresses here (one per line). First line should be the depot/start address."
    )
    addresses = [line.strip() for line in pasted.splitlines() if line.strip()]
else:
    uploaded = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
    if uploaded is not None:
        ext = uploaded.name.rsplit(".", 1)[-1].lower()
        try:
            if ext == "txt":
                addresses = _read_txt_lines(uploaded)
                if not addresses:
                    st.error("The uploaded TXT file is empty.")
            else:
                df = _read_csv_flex(uploaded)
                lower_cols = {c.lower(): c for c in df.columns}
                
                if "address" in lower_cols:
                    addresses = df[lower_cols["address"]].astype(str).tolist()
                else:
                    st.info("No single 'address' column detected.")
                    chosen = st.multiselect(
                        "Select 1+ columns to combine into an address (e.g., Street, City, State, ZIP):",
                        df.columns.tolist()
                    )
                    if chosen:
                        addresses = _combine_columns(df, chosen)
                
                if addresses:
                    # Optional: Add country if missing, to help geocoder
                    addresses = [
                        a if "USA" in a or "United States" in a else f"{a}, USA"
                        for a in addresses
                    ]
                    st.success(f"Loaded {len(addresses)} addresses from file.")
                else:
                    st.error("No valid addresses were found in the uploaded file.")
                    
        except Exception as e:
            st.error(f"Error reading file: {e}")

if addresses:
    st.success(f"Loaded {len(addresses)} addresses. (First is depot/start.)")
    if len(addresses) > 12:
        st.warning(
            "Large address count - geocoding and route-finding may be slow. "
            "Exact TSP will switch to a heuristic."
        )

# --- Route Computation ---

return_to_start = st.checkbox("Return to depot after route (round trip)", value=True)
compute_btn = st.button("Geocode & Compute Route")

if compute_btn:
    if not addresses:
        st.error("No addresses provided.")
    else:
        # 1. Geocoding
        with st.spinner("Geocoding addresses... (Nominatim)"):
            coords = []
            prog = st.progress(0)
            for i, addr in enumerate(addresses):
                coords.append(geocode_address(addr))
                prog.progress(int((i + 1) / max(1, len(addresses)) * 100))
                time.sleep(0.05) # Polite delay
        
        if any(c is None for c in coords):
            st.error("Some addresses failed to geocode:")
            for idx, (addr, c) in enumerate(zip(addresses, coords), start=1):
                if c is None:
                    st.write(f"{idx}: {addr}")
            st.info("Fix or remove those rows and run again.")
        else:
            n = len(coords)
            st.success("All addresses geocoded.")
            
            # 2. Distance Matrix
            with st.spinner("Building distance matrix..."):
                dist_matrix = build_distance_matrix(coords)
            
            # 3. Solve TSP
            if n <= 9:
                route, total_km = exact_tsp(dist_matrix, return_to_start=return_to_start)
                st.info("Used exact TSP (brute-force).")
            else:
                route, total_km = nearest_neighbor_tsp(dist_matrix, start=0, return_to_start=return_to_start)
                st.warning("Used nearest-neighbor heuristic for TSP (faster but not guaranteed optimal).")
            
            # 4. Display Results
            st.subheader("Route result")
            st.write(f"Route indices (0 = depot): {route}")
            st.write(f"Total route distance: **{total_km:.2f} km**")
            
            ordered = []
            for pos, idx in enumerate(route):
                ordered.append({
                    "order": pos + 1,
                    "index": idx,
                    "address": addresses[idx],
                    "lat": coords[idx][0],
                    "lon": coords[idx][1]
                })
            
            route_df = pd.DataFrame(ordered)
            st.dataframe(route_df)
            
            # Map
            map_df = pd.DataFrame(
                [(coords[idx][0], coords[idx][1]) for idx in route],
                columns=["lat", "lon"]
            )
            st.map(map_df)
            
            # 5. Drone Calculations
            st.subheader("Drone calculations and fleet estimate")
            
            drops = (n - 1) if return_to_start else (n - 1)
            # Avoid division by zero
            avg_drop_km = (total_km / max(1, drops))
            distance_per_drop_miles = avg_drop_km * 0.621371
            
            # Estimate times
            # Round trip time per drop in minutes
            round_trip_min = (distance_per_drop_miles * 2) / max(0.0001, speed_with_load) * 60.0
            handling_min = 10.0 # approx handling time per drop
            total_time_per_delivery_min = round_trip_min + handling_min
            
            # Deliveries per charge
            deliveries_per_charge = int(max(0, flight_time // total_time_per_delivery_min))
            
            if deliveries_per_charge <= 0:
                st.error("Battery not sufficient for a single delivery trip with current parameters.")
            else:
                workday_min = 8 * 60
                
                # Cycle time = (flight time per batch) + charge time
                # Here we simplify: one batch is 'deliveries_per_charge' drops
                cycle_total_time = (deliveries_per_charge * total_time_per_delivery_min) + charge_time
                
                if cycle_total_time > 0:
                    cycles_per_day = workday_min / cycle_total_time
                else:
                    cycles_per_day = 0
                
                deliveries_per_day = int(cycles_per_day * deliveries_per_charge)
                
                if deliveries_per_day > 0:
                    required_drones = math.ceil(drops / deliveries_per_day)
                else:
                    required_drones = math.inf
                
                # Metrics Output
                st.write(f" Estimated distance per drop: **{distance_per_drop_miles:.2f} miles**.")
                st.write(f" Round-trip flight time (per delivery): **{round_trip_min:.1f} min**.")
                st.write(f" Handling time per delivery (approximate): **{handling_min:.1f} min**.")
                st.write(f" Total time per delivery: **{total_time_per_delivery_min:.1f} min**.")
                st.write(f" Deliveries per charge (per drone): **{deliveries_per_charge}**.")
                st.write(f" Deliveries per day (per drone, 8 hours): **{deliveries_per_day}**.")
                
                if required_drones != math.inf:
                    st.write(f" Estimated required drones to finish all drops in one day: **{required_drones}**.")
                else:
                    st.write("- Estimated required drones: **N/A**.")

            # CSV Download
            csv = route_df.to_csv(index=False)
            st.download_button(
                "Download route as CSV",
                csv,
                file_name="route.csv",
                mime="text/csv"
            )

# --- Standard Boilerplate ---
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            # Code to run via subprocess if called directly with python
            this = os.path.abspath(__file__)
            subprocess.run([sys.executable, "-m", "streamlit", "run", this], check=True)
    except Exception:
        # Fallback
        pass
