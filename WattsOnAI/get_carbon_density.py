import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta, timezone
import state 

estimated_carbon_intensity = {
    'us-west-1': {
        "zone": "US-NW-PACW",
        "carbonIntensity": 123,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:26:07.868Z",
        "createdAt": "2025-10-14T17:11:49.394Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'us-west-2': {
        "zone": "US-NW-PACW",
        "carbonIntensity": 123,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:26:07.868Z",
        "createdAt": "2025-10-14T17:11:49.394Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'us-west1': {
        "zone": "US-NW-PACW",
        "carbonIntensity": 123,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:26:07.868Z",
        "createdAt": "2025-10-14T17:11:49.394Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'westus': {
        "zone": "US-CAL-CISO",
        "carbonIntensity": 67,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:27:00.822Z",
        "createdAt": "2025-10-14T17:11:37.643Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'westus2': {
        "zone": "US-NW-BPAT",
        "carbonIntensity": 97,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:26:47.337Z",
        "createdAt": "2025-10-14T17:11:43.342Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'westus3': {
        "zone": "US-SW-AZPS",
        "carbonIntensity": 245,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:26:38.064Z",
        "createdAt": "2025-10-14T17:11:20.159Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'us-east-1': {
        "zone": "US-MIDA-PJM",
        "carbonIntensity": 305,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-17T17:41:52.486Z",
        "createdAt": "2025-10-14T17:11:49.394Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'us-east-2': {
        "zone": "US-MIDA-PJM",
        "carbonIntensity": 299,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:27:00.822Z",
        "createdAt": "2025-10-14T17:11:49.394Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'us-east1': {
        "zone": "US-CAR-SCEG",
        "carbonIntensity": 333,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:26:21.760Z",
        "createdAt": "2025-10-14T17:11:37.643Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'us-east4': {
        "zone": "US-MIDA-PJM",
        "carbonIntensity": 299,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:27:00.822Z",
        "createdAt": "2025-10-14T17:11:49.394Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'eastus': {
        "zone": "US-MIDA-PJM",
        "carbonIntensity": 299,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:27:00.822Z",
        "createdAt": "2025-10-14T17:11:49.394Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'eastus2': {
        "zone": "US-MIDA-PJM",
        "carbonIntensity": 299,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:27:00.822Z",
        "createdAt": "2025-10-14T17:11:49.394Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'centralus': {
        "zone": "US-MIDW-MISO",
        "carbonIntensity": 325,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:26:06.246Z",
        "createdAt": "2025-10-14T17:11:20.159Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'uscentral1': {
        "zone": "US-CENT-SWPP",
        "carbonIntensity": 350,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:26:38.064Z",
        "createdAt": "2025-10-14T17:11:55.713Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'northcentralus': {
        "zone": "US-MIDW-MISO",
        "carbonIntensity": 325,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:26:06.246Z",
        "createdAt": "2025-10-14T17:11:20.159Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'southcentralus': {
        "zone": "US-TEX-ERCO",
        "carbonIntensity": 257,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T11:42:02.565Z",
        "createdAt": "2025-10-14T17:11:20.159Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    },
    'westcentralus': {
        "zone": "US-NW-PACE",
        "carbonIntensity": 187,
        "datetime": "2025-10-17T17:45:00.000Z",
        "updatedAt": "2025-10-18T13:26:22.926Z",
        "createdAt": "2025-10-14T17:11:55.713Z",
        "emissionFactorType": "direct",
        "temporalGranularity": "15_minutes"
    }
}

# Regional groupings for US cloud regions
us_west_regions = [
    'us-west-1',
    'us-west-2',
    'us-west1',
    'westus',
    'westus2',
    'westus3'
]

us_east_regions = [
    'us-east-1',
    'us-east-2',
    'us-east1',
    'us-east4',
    'eastus',
    'eastus2'
]

us_central_regions = [
    'centralus',
    'uscentral1',
    'northcentralus',
    'southcentralus',
    'westcentralus'
]


def get_current_carbon_intensity(username, password, latitude=None, longitude=None, cloud_region = None, use_WattTime=False, verbose=False):
    """
    Get current carbon intensity for a geographic location.
    
    Args:
        username: WattTime API username
        password: WattTime API password
        region: AWS/GCP/AZR Cloud Region
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        use_WattTime: If True, use WattTime API; if False, use Electricity Maps fallback data
        verbose: If True, print detailed information
    
    Returns:
        dict: Carbon intensity data with region, value, and units
    """
    
    # AS OF October 17, 2025
    # Fallback data from Electricity Maps (used when use_WattTime=False)
    # Pre-calculated carbon emissions from Electricity Maps API
    # Data Type: Carbon Intensity
    # Temporality: Latest
    # Datetime: 2025-10-17 20:22
    # Temporal Granularity: 15_minutes
    # Emission Factor Type: Direct
    
    # If use_WattTime is False, return Electricity Maps data
    if not use_WattTime:
        if verbose:
            print(f"\n{'='*70}")
            print(f"⚡ Using Electricity Maps Data")
            print(f"{'='*70}")
        
        if cloud_region and cloud_region in estimated_carbon_intensity.keys():
            em_data = estimated_carbon_intensity[cloud_region]
            
            # Convert gCO2eq/kWh to lbs CO2/MWh for consistency with WattTime
            # 1 gCO2/kWh = 0.0022046 lbs/kWh = 2.2046 lbs/MWh
            carbon_intensity_lbs_mwh = em_data['carbonIntensity'] * 2.2046
            
            if verbose:
                print(f"📍 Region: {cloud_region}")
                print(f"🗺️  Zone: {em_data['zone']}")
                print(f"📊 Carbon Intensity: {em_data['carbonIntensity']} gCO2eq/kWh")
                print(f"📊 Converted: {carbon_intensity_lbs_mwh:.2f} lbs CO2/MWh")
                print(f"🕒 Last Updated: {em_data['updatedAt']}")
                print(f"{'='*70}\n")
            
            return {
                "cloud_region": cloud_region, 
                "zone": em_data['zone'],
                "point_time": em_data['datetime'],
                "value": carbon_intensity_lbs_mwh,
                "units": "lbs_co2_per_mwh",
                "source": "ElectricityMaps",
                "original_value": em_data['carbonIntensity'],
                "original_units": "gCO2eq/kWh"
            }
        else:
            if verbose:
                print(f"⚠️  No Electricity Maps data for region: ({cloud_region})")
            raise ValueError(f"No Electricity Maps data available for region: {cloud_region}")
    
    
    # Validate latitude and longitude when using WattTime
    if latitude is None or longitude is None:
        if verbose:
            print(f"⚠️  Latitude and longitude are required when using WattTime API")
        raise ValueError("Latitude and longitude must be provided when using WattTime API")
    
    # WattTime API flow
    if verbose:
        print(f"\n{'='*70}")
        print(f"🌍 Fetching Carbon Intensity from WattTime API")
        print(f"{'='*70}")
        print(f"📍 Coordinates: Latitude={latitude:.6f}, Longitude={longitude:.6f}")
    
    try:
        # Step 1: Login to get token
        login_url = 'https://api.watttime.org/login'
        rsp = requests.get(login_url, auth=HTTPBasicAuth(username, password))
        rsp.raise_for_status()
        TOKEN = rsp.json()['token']
        
        if verbose:
            print(f"✅ Authentication successful")
        
        headers = {"Authorization": f"Bearer {TOKEN}"}
        
        # Step 2: Get region from lat/lon
        region_url = "https://api.watttime.org/v3/region-from-loc"
        region_params = {
            "latitude": str(latitude), 
            "longitude": str(longitude), 
            "signal_type": "co2_moer"
        }
        region_rsp = requests.get(region_url, headers=headers, params=region_params)
        region_rsp.raise_for_status()
        region = region_rsp.json()['region']
        
        if verbose:
            print(f"✅ Region identified: {region}")

        # Step 3: Get recent carbon intensity (past 10 minutes)
        # Fix: Use timezone-aware datetime
        now = datetime.now(timezone.utc)
        start_time = (now - timedelta(minutes=10)).replace(second=0, microsecond=0).isoformat()
        end_time = now.replace(second=0, microsecond=0).isoformat()

        hist_url = "https://api.watttime.org/v3/historical"
        hist_params = {
            "region": region,
            "start": start_time,
            "end": end_time,
            "signal_type": "co2_moer",
        }
        
        hist_rsp = requests.get(hist_url, headers=headers, params=hist_params)
        
        # Handle 403 Forbidden (subscription limitation)
        if hist_rsp.status_code == 403:
            if verbose:
                print(f"⚠️  Access denied to region '{region}' (subscription limitation)")
            return {
                "region": region,
                "latitude": latitude,
                "longitude": longitude,
                "point_time": None,
                "value": None,
                "units": "lbs_co2_per_mwh",
                "error": "Access denied - subscription limitation",
                "accessible": False,
                "source": "WattTime"
            }
        
        hist_rsp.raise_for_status()
        data = hist_rsp.json().get("data", [])

        if not data:
            if verbose:
                print(f"⚠️  No carbon intensity data available for region '{region}'")
            raise Exception("No carbon intensity data returned in the last 10 minutes.")

        latest_point = data[-1]
        
        result = {
            "region": region,
            "latitude": latitude,
            "longitude": longitude,
            "point_time": latest_point["point_time"],
            "value": latest_point["value"],
            "units": hist_rsp.json()["meta"]["units"],
            "accessible": True,
            "source": "WattTime"
        }
        
        if verbose:
            print(f"✅ Carbon intensity: {latest_point['value']} {result['units']}")
            print(f"{'='*70}\n")
        
        return result
        
    except requests.exceptions.HTTPError as e:
        if verbose:
            print(f"❌ HTTP Error: {e}")
        raise
    except Exception as e:
        if verbose:
            print(f"❌ Unexpected error: {e}")
        raise


def compute_carbon_emission(energy_joules, intensity_lbs_per_mwh, verbose=False):
    """
    Convert energy consumption to carbon emissions.
    
    Args:
        energy_joules: Energy consumption in Joules
        intensity_lbs_per_mwh: Carbon intensity in lbs CO2/MWh
        verbose: If True, print calculation details
    
    Returns:
        tuple: (total_lbs, total_kg) carbon emissions, or (None, None) if intensity is unavailable
    """
    if intensity_lbs_per_mwh is None:
        return None, None
    
    lbs_per_joule = intensity_lbs_per_mwh / 3.6e9
    total_lbs = energy_joules * lbs_per_joule
    total_kg = total_lbs * 0.453592
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"💨 Carbon Emission Calculation")
        print(f"{'='*70}")
        print(f"⚡ Energy Consumption: {energy_joules:,.2f} J ({energy_joules/3.6e6:.2f} kWh)")
        print(f"🌡️  Carbon Intensity: {intensity_lbs_per_mwh:.2f} lbs CO2/MWh")
        print(f"💨 Total Emissions: {total_lbs:.4f} lbs CO2 ({total_kg:.4f} kg CO2)")
        print(f"{'='*70}\n")
    
    return total_lbs, total_kg