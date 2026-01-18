"""
REEF DIGITAL TWIN - PROFESSIONAL DASHBOARD v8
==============================================
- Light theme (readable)
- Multi-sensor plots (all sensors on same chart)
- 4 sensor plots: pH, Temp, Ammonia, ORP
- 4 more sensor plots: Salinity, PUR, Conductivity, TDS
- 4 lab data plots: Ca, Mg, Alkalinity, Nitrate
- Chat-controlled model runs
- Secure API key loading from .env file or environment variables

Author: Manel Garrido-Baserba
Date: January 2025
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import subprocess
import time
import numpy as np
from anthropic import Anthropic
from report_generator import generate_pdf_report
from pathlib import Path

# ==============================================================================
# LOAD ENVIRONMENT VARIABLES (for API keys)
# ==============================================================================
# Try to load from .env file if it exists (for local development)
try:
    from dotenv import load_dotenv
    # Look for .env in the same directory as this script
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env from {env_path}")
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Digital Twin Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

AQUAPI_URL = "https://aquapi-db.workstation1606.dev/sensors/all"
AQUAPI_HEADERS = {
    "CF-Access-Client-Id": "a2fa58ac4f021069e0720892b6a53e74.access",
    "CF-Access-Client-Secret": "45b7a186f08ad41915c74617c811c9450c9cfd7d02f4545e761d6a3394213404",
    "Content-Type": "application/json"
}

PREDICTIONS_HISTORY_PATH = r"C:\Users\MGarrido\OneDrive - Brown and Caldwell\WATER\DSS_DT\ReefDigitalTwin\outputs\predictions_history.csv"
PREDICTIONS_LATEST_PATH = r"C:\Users\MGarrido\OneDrive - Brown and Caldwell\WATER\DSS_DT\ReefDigitalTwin\outputs\predictions_latest.json"
TIMESERIES_PATH = r"C:\Users\MGarrido\OneDrive - Brown and Caldwell\WATER\DSS_DT\ReefDigitalTwin\outputs\timeseries_latest.json"
SENSITIVITY_PATH = r"C:\Users\MGarrido\OneDrive - Brown and Caldwell\WATER\DSS_DT\ReefDigitalTwin\outputs\sensitivity_results.json"
PROGRESS_PATH = r"C:\Users\MGarrido\OneDrive - Brown and Caldwell\WATER\DSS_DT\ReefDigitalTwin\outputs\model_progress.json"
LAB_DATA_PATH = r"C:\Users\MGarrido\OneDrive - Brown and Caldwell\WATER\DSS_DT\Research Station\Parameters_Tracking.xlsx"
SUMO_SCRIPT_PATH = r"C:\Users\MGarrido\OneDrive - Brown and Caldwell\WATER\DSS_DT\ReefDigitalTwin\scripts\reef_sumo_runner_v5.py"
SENSITIVITY_SCRIPT_PATH = r"C:\Users\MGarrido\OneDrive - Brown and Caldwell\WATER\DSS_DT\ReefDigitalTwin\scripts\reef_sensitivity_runner.py"
SUMO_WORKING_DIR = r"C:\Users\MGarrido\OneDrive - Brown and Caldwell\WATER\DSS_DT\SUMO\POND_SUMO"
PYTHON_PATH = r"C:\Users\MGarrido\AppData\Local\Programs\Python\Python312\python.exe"

PH_OFFSET = -0.3

# ==============================================================================
# DATA FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=300)
def fetch_sensor_data():
    """Fetch all sensor data from AquaPi API"""
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(AQUAPI_URL, headers=AQUAPI_HEADERS, timeout=30, verify=False)
        response.raise_for_status()
        data = response.json()
        readings = data.get("readings", [])
        df = pd.DataFrame(readings)
        if not df.empty:
            df['real_timestamp'] = pd.to_datetime(df['real_timestamp'])
            df = df.sort_values('real_timestamp', ascending=False)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=3600)
def load_lab_data():
    """Load lab data from Excel"""
    try:
        if os.path.exists(LAB_DATA_PATH):
            df = pd.read_excel(LAB_DATA_PATH, sheet_name='LAB_DATA')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date', ascending=True)  # Chronological for plots
            return df, None
        return pd.DataFrame(), "File not found"
    except Exception as e:
        return pd.DataFrame(), str(e)


def load_predictions_history():
    """Load model predictions"""
    try:
        if os.path.exists(PREDICTIONS_HISTORY_PATH):
            df = pd.read_csv(PREDICTIONS_HISTORY_PATH)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for col in ['sim_tss_sump', 'sim_TSS_effluent']:
                if col in df.columns:
                    df = df[df[col] < 1000000]
                    break
            return df.sort_values('timestamp', ascending=False), None
        return pd.DataFrame(), "File not found"
    except Exception as e:
        return pd.DataFrame(), str(e)


def load_latest_prediction():
    """Load most recent prediction"""
    try:
        if os.path.exists(PREDICTIONS_LATEST_PATH):
            with open(PREDICTIONS_LATEST_PATH, 'r') as f:
                return json.load(f), None
        return {}, "File not found"
    except Exception as e:
        return {}, str(e)


def load_timeseries():
    """Load time series data from latest simulation"""
    try:
        if os.path.exists(TIMESERIES_PATH):
            with open(TIMESERIES_PATH, 'r') as f:
                return json.load(f), None
        return {}, "File not found"
    except Exception as e:
        return {}, str(e)


def load_progress():
    """Load current model progress"""
    try:
        if os.path.exists(PROGRESS_PATH):
            with open(PROGRESS_PATH, 'r') as f:
                return json.load(f)
        return None
    except:
        return None


def load_sensitivity_results():
    """Load sensitivity analysis results"""
    try:
        if os.path.exists(SENSITIVITY_PATH):
            with open(SENSITIVITY_PATH, 'r') as f:
                return json.load(f), None
        return {}, "File not found"
    except Exception as e:
        return {}, str(e)


def process_sensor_data(df):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    seneye = df[df['sensor_id'] == 'SUD-1'].copy()
    kactoily1 = df[df['sensor_id'] == 'SUD-2'].copy()
    kactoily2 = df[df['sensor_id'] == 'SUD-3'].copy()
    return seneye, kactoily1, kactoily2


def calculate_total_ammonia(free_nh3, ph, temp_c):
    """
    Convert free ammonia (NH3) to total ammonia (NH3 + NH4+)
    Based on pH and temperature using the ammonia equilibrium equation
    """
    if free_nh3 is None or ph is None or temp_c is None:
        return None
    if free_nh3 == 0:
        return 0
    
    temp_k = temp_c + 273.15
    pKa = 0.09018 + (2729.92 / temp_k)
    fraction_nh3 = 1 / (1 + pow(10, pKa - ph))
    
    if fraction_nh3 > 0:
        return free_nh3 / fraction_nh3
    return 0


def get_latest_values(seneye, kactoily1, kactoily2):
    latest = {}
    if not seneye.empty:
        row = seneye.iloc[0]
        latest['seneye_temp'] = row.get('temperature_c')
        latest['seneye_ph_raw'] = row.get('ph')
        latest['seneye_ph'] = row.get('ph', 0) + PH_OFFSET if row.get('ph') else None
        latest['seneye_ammonia'] = row.get('ammonia_mg_l')
        latest['seneye_par'] = row.get('par')
        latest['seneye_pur'] = row.get('pur')
        latest['seneye_time'] = row.get('real_timestamp')
        
        # Calculate total ammonia equivalent
        latest['total_ammonia'] = calculate_total_ammonia(
            latest['seneye_ammonia'],
            latest['seneye_ph'],
            latest['seneye_temp']
        )
    if not kactoily1.empty:
        row = kactoily1.iloc[0]
        latest['k1_temp'] = row.get('temperature_c')
        latest['k1_ph'] = row.get('ph')
        latest['k1_orp'] = row.get('orp')
        latest['k1_salinity'] = row.get('salinity')
        latest['k1_conductivity'] = row.get('electrical_conductivity')
        latest['k1_tds'] = row.get('tds')
        latest['k1_time'] = row.get('real_timestamp')
    if not kactoily2.empty:
        row = kactoily2.iloc[0]
        latest['k2_temp'] = row.get('temperature_c')
        latest['k2_ph'] = row.get('ph')
        latest['k2_orp'] = row.get('orp')
        latest['k2_salinity'] = row.get('salinity')
        latest['k2_conductivity'] = row.get('electrical_conductivity')
        latest['k2_tds'] = row.get('tds')
        latest['k2_time'] = row.get('real_timestamp')
    return latest


def get_latest_lab(lab_df):
    if lab_df.empty:
        return {}
    row = lab_df.iloc[-1]  # Last row (most recent)
    return {
        'date': row.get('Date'),
        'ph': row.get('pH'),
        'ammonia': row.get('Ammonia (ppm)'),
        'nitrate': row.get('Nitrate (ppm)'),
        'phosphate': row.get('Phosphate (ppm)'),
        'calcium': row.get('Calcium (ppm)'),
        'magnesium': row.get('Magnesium (ppm)'),
        'alkalinity': row.get('Alkalinity (dKH)')
    }


# ==============================================================================
# MODEL EXECUTION
# ==============================================================================

def clear_progress():
    """Clear progress file"""
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)


def run_sumo_model():
    """Run SUMO model (basic version for compatibility)"""
    try:
        clear_progress()
        result = subprocess.run(
            [PYTHON_PATH, SUMO_SCRIPT_PATH],
            cwd=SUMO_WORKING_DIR,
            capture_output=True,
            text=True,
            timeout=180
        )
        if result.returncode == 0:
            if os.path.exists(PREDICTIONS_LATEST_PATH):
                with open(PREDICTIONS_LATEST_PATH, 'r') as f:
                    return True, json.load(f), result.stdout
            return True, {}, result.stdout
        return False, {}, f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, {}, "Model timed out"
    except Exception as e:
        return False, {}, str(e)


def run_sumo_with_progress(chat_container):
    """Run SUMO model with live progress updates in chat"""
    import time
    
    clear_progress()
    
    # Start the process
    process = subprocess.Popen(
        [PYTHON_PATH, SUMO_SCRIPT_PATH],
        cwd=SUMO_WORKING_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Create progress display in chat
    progress_messages = []
    last_pct = -1
    
    while process.poll() is None:
        progress = load_progress()
        if progress:
            pct = progress.get('progress', 0)
            msg = progress.get('message', 'Running...')
            status = progress.get('status', 'running')
            
            if pct != last_pct:
                # Create progress bar
                bar_len = 20
                filled = int(bar_len * pct / 100)
                bar = '█' * filled + '░' * (bar_len - filled)
                progress_text = f"`[{bar}]` **{pct:.0f}%** - {msg}"
                progress_messages.append(progress_text)
                last_pct = pct
        
        time.sleep(0.3)
    
    # Get final result
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        if os.path.exists(PREDICTIONS_LATEST_PATH):
            with open(PREDICTIONS_LATEST_PATH, 'r') as f:
                return True, json.load(f), progress_messages
        return True, {}, progress_messages
    return False, {}, progress_messages


def run_sensitivity_analysis(parameter, scenarios=None):
    """Run sensitivity analysis on a parameter"""
    try:
        if scenarios is None:
            scenarios = [-30, -15, 0, 15, 30]  # Default: ±30%, ±15%, baseline
        
        scenarios_str = ','.join(map(str, scenarios))
        
        result = subprocess.run(
            [PYTHON_PATH, SENSITIVITY_SCRIPT_PATH, '--param', parameter, '--scenarios', scenarios_str],
            cwd=SUMO_WORKING_DIR,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout for multiple runs
        )
        
        if result.returncode == 0:
            if os.path.exists(SENSITIVITY_PATH):
                with open(SENSITIVITY_PATH, 'r') as f:
                    return True, json.load(f), result.stdout
            return True, {}, result.stdout
        return False, {}, f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, {}, "Sensitivity analysis timed out"
    except Exception as e:
        return False, {}, str(e)


def format_model_results(pred):
    if not pred:
        return "No results."
    return f"""
**✅ Model Run Complete**

| Metric | Value |
|--------|-------|
| Algae Pond 1 | {pred.get('sim_algae_pond1', 0):.1f} g COD/m³ |
| Algae Pond 2 | {pred.get('sim_algae_pond2', 0):.1f} g COD/m³ |
| NH4 Effluent | {pred.get('sim_nh4_effluent', 0):.1f} g N/m³ |
| pH Pond 1 | {pred.get('sim_ph_pond1', 0):.2f} |

**Performance:** NH4 Removal **{pred.get('calc_nh4_removal_pct', 0):.1f}%** | TN Removal **{pred.get('calc_TN_removal_pct', 0):.1f}%**
"""


def format_sensitivity_results(sens_data):
    """Format sensitivity analysis results for chat display"""
    if not sens_data:
        return "No sensitivity results available."
    
    param = sens_data.get('parameter', 'Unknown')
    scenarios = sens_data.get('scenarios', [])
    
    result = f"**📊 Sensitivity Analysis: {param.upper()}**\n\n"
    result += "| Scenario | NH₄ Removal | Algae | pH |\n"
    result += "|----------|-------------|-------|----|\n"
    
    for s in scenarios:
        pct = s.get('pct_change', 0)
        label = f"+{pct}%" if pct > 0 else f"{pct}%" if pct < 0 else "Baseline"
        nh4 = s.get('nh4_removal_pct', 0)
        algae = s.get('algae_pond1', 0)
        ph = s.get('ph_pond1', 0)
        result += f"| {label} | {nh4:.1f}% | {algae:.1f} | {ph:.2f} |\n"
    
    result += "\n*View plots in the bottom section*"
    return result


def detect_sensitivity_request(question):
    """Detect if user is asking for sensitivity analysis"""
    q = question.lower()
    
    # Check for sensitivity keywords
    sens_keywords = ['sensitivity', 'what if', 'scenario', 'impact of', 'effect of', 'change in', 'vary', 'increase', 'decrease']
    has_sens_keyword = any(kw in q for kw in sens_keywords)
    
    # Check for parameter keywords
    param_map = {
        'light': ['light', 'solar', 'radiation', 'sun', 'irradiance'],
        'temperature': ['temperature', 'temp', 'heat', 'warm', 'cold'],
        'tkn': ['tkn', 'nitrogen', 'ammonia', 'nh4', 'loading', 'influent'],
        'flow': ['flow', 'hrt', 'retention', 'volume'],
        'co2': ['co2', 'carbon dioxide', 'carbon']
    }
    
    detected_param = None
    for param, keywords in param_map.items():
        if any(kw in q for kw in keywords):
            detected_param = param
            break
    
    if has_sens_keyword and detected_param:
        return detected_param
    
    # Check for explicit sensitivity request
    if 'sensitivity' in q and 'run' in q:
        # Try to extract parameter from question
        for param, keywords in param_map.items():
            if any(kw in q for kw in keywords):
                return param
        return 'light'  # Default parameter
    
    return None


def create_sensitivity_timeseries_plot(sens_data, var_name, title):
    """Create time series plot with multiple scenario lines"""
    if not sens_data or 'scenarios' not in sens_data:
        return None
    
    scenarios = sens_data.get('scenarios', [])
    if not scenarios:
        return None
    
    fig = go.Figure()
    
    # Color palette for scenarios
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    line_styles = ['dash', 'dot', 'solid', 'dot', 'dash']
    
    for i, scenario in enumerate(scenarios):
        pct = scenario.get('pct_change', 0)
        label = f"+{pct}%" if pct > 0 else f"{pct}%" if pct < 0 else "Baseline"
        
        timeseries = scenario.get('timeseries', {})
        if var_name in timeseries and 'time_days' in timeseries:
            t = timeseries['time_days']
            y = timeseries[var_name]
            
            # Make baseline line thicker and solid
            width = 3 if pct == 0 else 2
            dash = None if pct == 0 else line_styles[i % len(line_styles)]
            
            fig.add_trace(go.Scatter(
                x=t, y=y,
                mode='lines',
                name=label,
                line=dict(color=colors[i % len(colors)], width=width, dash=dash)
            ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Time (days)",
        height=280,
        margin=dict(l=10, r=10, t=40, b=50),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=9)
        )
    )
    return fig


# ==============================================================================
# CHART FUNCTIONS
# ==============================================================================

def create_multi_sensor_plot(seneye, kactoily1, kactoily2, column, title, unit=""):
    """Create a plot with multiple sensors on same chart"""
    fig = go.Figure()
    
    colors = {'Seneye': '#1f77b4', 'Display': '#2ca02c', 'Sump': '#ff7f0e'}
    
    # Add Seneye data
    if not seneye.empty and column in seneye.columns:
        data = seneye[seneye[column].notna()]
        if not data.empty:
            fig.add_trace(go.Scatter(
                x=data['real_timestamp'],
                y=data[column],
                mode='lines',
                name='Seneye',
                line=dict(color=colors['Seneye'], width=2)
            ))
    
    # Add Kactoily1 (Display) data
    if not kactoily1.empty and column in kactoily1.columns:
        data = kactoily1[kactoily1[column].notna()]
        if not data.empty:
            fig.add_trace(go.Scatter(
                x=data['real_timestamp'],
                y=data[column],
                mode='lines',
                name='Display',
                line=dict(color=colors['Display'], width=2)
            ))
    
    # Add Kactoily2 (Sump) data
    if not kactoily2.empty and column in kactoily2.columns:
        data = kactoily2[kactoily2[column].notna()]
        if not data.empty:
            fig.add_trace(go.Scatter(
                x=data['real_timestamp'],
                y=data[column],
                mode='lines',
                name='Sump',
                line=dict(color=colors['Sump'], width=2)
            ))
    
    fig.update_layout(
        title=f"{title} {unit}",
        height=220,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig


def create_single_sensor_plot(df, column, title, color='#1f77b4'):
    """Create a simple single-sensor plot"""
    if df.empty or column not in df.columns:
        return None
    
    data = df[df[column].notna()]
    if data.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['real_timestamp'],
        y=data[column],
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba(31, 119, 180, 0.1)'
    ))
    
    fig.update_layout(
        title=title,
        height=220,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig


def create_lab_plot(lab_df, column, title, color='#d62728'):
    """Create a plot for lab data over time"""
    if lab_df.empty or column not in lab_df.columns:
        return None
    
    data = lab_df[lab_df[column].notna()]
    if data.empty:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data[column],
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=title,
        height=220,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig


def create_timeseries_plot(ts_data, var_name, title, color='#1f77b4', var_name2=None, color2='#2ca02c'):
    """Create a time series plot from simulation data"""
    if not ts_data or 'time_days' not in ts_data or 'series' not in ts_data:
        return None
    
    if var_name not in ts_data['series']:
        return None
    
    t = ts_data['time_days']
    y = ts_data['series'][var_name]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=y,
        mode='lines',
        name=var_name.replace('_', ' ').title(),
        line=dict(color=color, width=2)
    ))
    
    if var_name2 and var_name2 in ts_data['series']:
        y2 = ts_data['series'][var_name2]
        fig.add_trace(go.Scatter(
            x=t, y=y2,
            mode='lines',
            name=var_name2.replace('_', ' ').title(),
            line=dict(color=color2, width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (days)",
        height=200,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_sankey_diagram(latest_pred):
    """Create nitrogen mass balance Sankey diagram"""
    if not latest_pred:
        return None
    
    # Get values from model prediction
    influent_n = latest_pred.get('model_TKN_gm3', 34.4)
    pond1_n = latest_pred.get('sim_nh4_pond1', 14)
    pond2_n = latest_pred.get('sim_nh4_pond2', 12)
    effluent_n = latest_pred.get('sim_nh4_effluent', 9)
    
    # Calculate removals (what was taken up/converted)
    removed_pond1 = max(0, influent_n - pond1_n)
    removed_pond2 = max(0, pond1_n - pond2_n)
    removed_total = max(0, pond2_n - effluent_n)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Influent N", "Pond 1", "Pond 2", "Effluent", "Algae Uptake P1", "Algae Uptake P2", "Final Removal"],
            color=["#d62728", "#ff9896", "#ffbb78", "#2ca02c", "#98df8a", "#98df8a", "#98df8a"]
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 2],
            target=[1, 4, 2, 5, 3, 6],
            value=[pond1_n, removed_pond1, pond2_n, removed_pond2, effluent_n, removed_total],
            color=["rgba(255, 152, 150, 0.5)", "rgba(152, 223, 138, 0.5)", 
                   "rgba(255, 187, 120, 0.5)", "rgba(152, 223, 138, 0.5)",
                   "rgba(44, 160, 44, 0.5)", "rgba(152, 223, 138, 0.5)"]
        )
    )])
    
    fig.update_layout(
        title="Nitrogen Mass Balance (g N/m³)",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=11)
    )
    return fig


def create_radar_chart(latest_pred, latest_sensors):
    """Create performance radar chart comparing current vs targets"""
    if not latest_pred:
        return None
    
    # Define metrics and targets
    categories = ['NH₄ Removal', 'TN Removal', 'Algae Growth', 'pH Stability', 'DO Level']
    
    # Current values (normalized to 0-100 scale)
    nh4_removal = latest_pred.get('calc_nh4_removal_pct', 0)
    tn_removal = latest_pred.get('calc_TN_removal_pct', 0)
    
    # Algae: target ~100 g/m³, scale 0-200 to 0-100
    algae = min(100, latest_pred.get('sim_algae_pond1', 0) / 2 * 100 / 100)
    
    # pH: target 8.0-8.5, penalize deviation
    ph_val = latest_pred.get('sim_ph_pond1', 8.0)
    ph_score = max(0, 100 - abs(ph_val - 8.2) * 20)
    
    # DO: target >8 mg/L, scale 0-15 to 0-100
    do_val = latest_pred.get('sim_DO_pond1', 10)
    do_score = min(100, do_val / 12 * 100)
    
    current_values = [nh4_removal, tn_removal, algae, ph_score, do_score]
    target_values = [80, 60, 80, 90, 80]  # Target performance
    
    fig = go.Figure()
    
    # Target area
    fig.add_trace(go.Scatterpolar(
        r=target_values + [target_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(color='#2ca02c', width=2, dash='dash'),
        name='Target'
    ))
    
    # Current performance
    fig.add_trace(go.Scatterpolar(
        r=current_values + [current_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.3)',
        line=dict(color='#1f77b4', width=2),
        name='Current'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title="Performance vs Targets",
        height=300,
        margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig


# ==============================================================================
# CLAUDE CHAT
# ==============================================================================

def get_claude_client():
    """
    Get Claude API client with key from multiple sources (in priority order):
    1. Streamlit secrets (for cloud deployment)
    2. Environment variable ANTHROPIC_API_KEY (from .env file or system)
    3. Session state (manual entry in sidebar)
    """
    api_key = None
    
    # 1. Try Streamlit secrets first (for cloud deployment)
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
        if api_key:
            return Anthropic(api_key=api_key)
    except Exception:
        pass  # Secrets not configured
    
    # 2. Try environment variable (from .env or system)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return Anthropic(api_key=api_key)
    
    # 3. Try session state (manual entry)
    api_key = st.session_state.get("api_key")
    if api_key:
        return Anthropic(api_key=api_key)
    
    return None


def get_api_key_status():
    """Return status message about API key configuration"""
    try:
        if st.secrets.get("ANTHROPIC_API_KEY"):
            return "✅ API Key: Streamlit Secrets"
    except:
        pass
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "✅ API Key: Environment Variable"
    
    if st.session_state.get("api_key"):
        return "✅ API Key: Manual Entry"
    
    return "⚠️ No API Key configured"


def prepare_data_context(latest_sensors, latest_pred, latest_lab, predictions_df):
    # Pre-format values to avoid f-string issues
    temp_val = latest_sensors.get('seneye_temp')
    temp_str = f"{temp_val:.1f}" if temp_val else "N/A"
    
    ph_val = latest_sensors.get('seneye_ph')
    ph_str = f"{ph_val:.2f}" if ph_val else "N/A"
    
    nh3_val = latest_sensors.get('seneye_ammonia')
    nh3_str = f"{nh3_val:.3f}" if nh3_val else "N/A"
    
    orp_val = latest_sensors.get('k1_orp')
    orp_str = f"{orp_val:.0f}" if orp_val else "N/A"
    
    alk_val = latest_lab.get('alkalinity') if latest_lab else None
    alk_str = f"{alk_val:.1f}" if alk_val else "N/A"
    
    ca_val = latest_lab.get('calcium') if latest_lab else None
    ca_str = f"{ca_val:.0f}" if ca_val else "N/A"
    
    mg_val = latest_lab.get('magnesium') if latest_lab else None
    mg_str = f"{mg_val:.0f}" if mg_val else "N/A"
    
    no3_val = latest_lab.get('nitrate') if latest_lab else None
    no3_str = f"{no3_val:.1f}" if no3_val else "N/A"
    
    lab_date = latest_lab.get('date', 'N/A') if latest_lab else 'N/A'
    
    ctx = f"""
LIVE SENSORS ({datetime.now().strftime('%H:%M')}):
- Temperature: {temp_str}°C
- pH: {ph_str}
- Ammonia: {nh3_str} mg/L
- ORP: {orp_str} mV

LAB DATA (last: {lab_date}):
- Alkalinity: {alk_str} dKH
- Calcium: {ca_str} ppm
- Magnesium: {mg_str} ppm
- Nitrate: {no3_str} ppm
"""
    if latest_pred:
        algae_val = latest_pred.get('sim_algae_pond1', 0)
        nh4_rem = latest_pred.get('calc_nh4_removal_pct', 0)
        tn_rem = latest_pred.get('calc_TN_removal_pct', 0)
        ctx += f"""
MODEL PREDICTION:
- Algae: {algae_val:.1f} g COD/m³
- NH4 Removal: {nh4_rem:.1f}%
- TN Removal: {tn_rem:.1f}%
"""
    return ctx


def detect_model_request(q):
    triggers = ["run the model", "run model", "run simulation", "run sumo", "execute", "trigger"]
    return any(t in q.lower() for t in triggers)


def chat_with_claude(question, context, history):
    """Chat with Claude - returns (response, action_type, action_param)
    action_type: None, 'model', 'sensitivity'
    """
    client = get_claude_client()
    if not client:
        return "⚠️ **No API key configured.** Create a `.env` file with `ANTHROPIC_API_KEY=sk-ant-...` or enter it in Settings.", None, None
    
    # Check for model run request
    if detect_model_request(question):
        return None, 'model', None
    
    # Check for sensitivity analysis request
    sens_param = detect_sensitivity_request(question)
    if sens_param:
        return None, 'sensitivity', sens_param
    
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": question})
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=f"""You are an AI for a nature-based treatment digital twin. Be concise, use numbers.
            
You can help with:
- Running the SUMO model (say "run the model")
- Sensitivity analysis (say "run sensitivity on light/temperature/tkn/flow")
- Explaining system performance
- Answering questions about the data

DATA:
{context}""",
            messages=messages
        )
        return response.content[0].text, None, None
    except Exception as e:
        return f"Error: {e}", None, None


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "sensitivity_results" not in st.session_state:
        st.session_state.sensitivity_results = None
    
    # Load all data
    sensor_df, sensor_err = fetch_sensor_data()
    lab_df, lab_err = load_lab_data()
    predictions_df, _ = load_predictions_history()
    latest_pred, _ = load_latest_prediction()
    sens_data, _ = load_sensitivity_results()  # Load sensitivity results if available
    
    # Update session state if file exists
    if sens_data:
        st.session_state.sensitivity_results = sens_data
    
    seneye, kactoily1, kactoily2 = process_sensor_data(sensor_df)
    latest_sensors = get_latest_values(seneye, kactoily1, kactoily2)
    latest_lab = get_latest_lab(lab_df)
    
    # Filter to recent data (7 days)
    cutoff = datetime.now() - timedelta(days=7)
    if not seneye.empty:
        seneye_recent = seneye[seneye['real_timestamp'] > cutoff].copy()
        # Calculate total ammonia equivalent for the time series
        seneye_recent['total_ammonia'] = seneye_recent.apply(
            lambda row: calculate_total_ammonia(
                row.get('ammonia_mg_l'),
                (row.get('ph', 0) + PH_OFFSET) if row.get('ph') else None,
                row.get('temperature_c')
            ),
            axis=1
        )
    else:
        seneye_recent = pd.DataFrame()
    if not kactoily1.empty:
        kactoily1_recent = kactoily1[kactoily1['real_timestamp'] > cutoff]
    else:
        kactoily1_recent = pd.DataFrame()
    if not kactoily2.empty:
        kactoily2_recent = kactoily2[kactoily2['real_timestamp'] > cutoff]
    else:
        kactoily2_recent = pd.DataFrame()
    
    # ======================== SIDEBAR ========================
    with st.sidebar:
        # API Key Settings - shows status and allows manual entry as fallback
        with st.expander("⚙️ Settings", expanded=False):
            # Show current API key status
            st.markdown(get_api_key_status())
            
            # Only show manual entry if no env var is set
            if not os.environ.get("ANTHROPIC_API_KEY"):
                st.caption("Or enter manually:")
                api_key = st.text_input("Anthropic API Key", value=st.session_state.api_key, type="password", label_visibility="collapsed")
                if api_key:
                    st.session_state.api_key = api_key
            else:
                st.caption("Using .env file or system variable")
        
        st.markdown("---")
        
        # Model Control
        st.markdown("## 🔬 Model Control")
        if st.button("▶️ Run SUMO Model", use_container_width=True, type="primary"):
            clear_progress()
            
            with st.status("🔄 Running SUMO simulation...", expanded=True) as status:
                st.write("Starting model...")
                
                # Start subprocess
                process = subprocess.Popen(
                    [PYTHON_PATH, SUMO_SCRIPT_PATH],
                    cwd=SUMO_WORKING_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Poll for progress
                last_pct = -1
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while process.poll() is None:
                    progress = load_progress()
                    if progress:
                        pct = progress.get('progress', 0)
                        msg = progress.get('message', 'Running...')
                        
                        if pct != last_pct:
                            progress_bar.progress(int(pct))
                            status_text.write(f"**{pct:.0f}%** - {msg}")
                            last_pct = pct
                    
                    time.sleep(0.2)
                
                # Get final result
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    progress_bar.progress(100)
                    status.update(label="✅ Model complete!", state="complete", expanded=False)
                    
                    if os.path.exists(PREDICTIONS_LATEST_PATH):
                        with open(PREDICTIONS_LATEST_PATH, 'r') as f:
                            pred = json.load(f)
                        st.session_state.chat_history.append({"role": "assistant", "content": format_model_results(pred)})
                else:
                    status.update(label="❌ Model failed", state="error")
                    st.error(stderr[:200] if stderr else "Unknown error")
            
            st.rerun()
        
        if latest_pred:
            st.caption(f"Last: {latest_pred.get('timestamp', 'N/A')[:16]}")
        
        st.markdown("---")
        
        # Report Generator - PDF
        st.markdown("## 📋 Reports")
        report_type = st.selectbox("Report Type", ["Daily Summary", "Weekly Analysis"], label_visibility="collapsed")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("📄 PDF", use_container_width=True, type="primary"):
                api_key = os.environ.get("ANTHROPIC_API_KEY") or st.session_state.get("api_key")
                if not api_key:
                    st.error("Enter API key in Settings")
                else:
                    with st.spinner("Generating PDF report..."):
                        try:
                            reports_dir = r"C:\Users\MGarrido\OneDrive - Brown and Caldwell\WATER\DSS_DT\ReefDigitalTwin\reports"
                            os.makedirs(reports_dir, exist_ok=True)
                            report_path = os.path.join(reports_dir, f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                            
                            generate_pdf_report(
                                output_path=report_path,
                                sensor_df=sensor_df,
                                predictions_df=predictions_df,
                                lab_df=lab_df,
                                api_key=api_key,
                                report_type="Daily" if "Daily" in report_type else "Weekly"
                            )
                            
                            with open(report_path, "rb") as f:
                                st.session_state['report_bytes'] = f.read()
                            st.session_state['report_name'] = os.path.basename(report_path)
                            st.success("✅ PDF Ready!")
                        except Exception as e:
                            st.error(f"Error: {str(e)[:80]}")
                    st.rerun()
        
        with col_r2:
            if 'report_bytes' in st.session_state:
                st.download_button("⬇️ Download", st.session_state['report_bytes'], 
                                   st.session_state['report_name'], "application/pdf",
                                   use_container_width=True)
        
        st.markdown("---")
        
        # Lab Data Section - Compact with smaller values
        st.markdown("## 🧪 Lab Data")
        if latest_lab:
            # Use custom HTML for smaller, compact display
            alk = latest_lab.get('alkalinity')
            ca = latest_lab.get('calcium')
            mg = latest_lab.get('magnesium')
            no3 = latest_lab.get('nitrate')
            po4 = latest_lab.get('phosphate')
            
            alk_str = f"{alk:.1f}" if alk else "—"
            ca_str = f"{ca:.0f}" if ca else "—"
            mg_str = f"{mg:.0f}" if mg else "—"
            no3_str = f"{no3:.1f}" if no3 else "—"
            po4_str = f"{po4:.2f}" if po4 else "—"
            
            st.markdown(f"""
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <div style='text-align: center; flex: 1;'>
                    <div style='font-size: 11px; color: #666;'>Alk</div>
                    <div style='font-size: 20px; font-weight: bold;'>{alk_str}</div>
                    <div style='font-size: 10px; color: #22c55e;'>↑ dKH</div>
                </div>
                <div style='text-align: center; flex: 1;'>
                    <div style='font-size: 11px; color: #666;'>Ca</div>
                    <div style='font-size: 20px; font-weight: bold;'>{ca_str}</div>
                    <div style='font-size: 10px; color: #22c55e;'>↑ ppm</div>
                </div>
                <div style='text-align: center; flex: 1;'>
                    <div style='font-size: 11px; color: #666;'>Mg</div>
                    <div style='font-size: 20px; font-weight: bold;'>{mg_str}</div>
                    <div style='font-size: 10px; color: #22c55e;'>↑ ppm</div>
                </div>
            </div>
            <div style='display: flex; justify-content: space-around; margin-bottom: 8px;'>
                <div style='text-align: center; flex: 1;'>
                    <div style='font-size: 11px; color: #666;'>NO₃</div>
                    <div style='font-size: 20px; font-weight: bold;'>{no3_str}</div>
                    <div style='font-size: 10px; color: #22c55e;'>↑ ppm</div>
                </div>
                <div style='text-align: center; flex: 1;'>
                    <div style='font-size: 11px; color: #666;'>PO₄</div>
                    <div style='font-size: 20px; font-weight: bold;'>{po4_str}</div>
                    <div style='font-size: 10px; color: #22c55e;'>↑ ppm</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"Tested: {latest_lab.get('date').strftime('%b %d') if latest_lab.get('date') else 'N/A'}")
        else:
            st.warning("No lab data")
        
        st.markdown("---")
        
        # Sensor Status
        st.markdown("## 📡 Sensors")
        
        def sensor_status(name, time_val):
            if time_val is None:
                return f"🔴 {name}"
            age = (datetime.now() - pd.to_datetime(time_val)).total_seconds() / 60
            if age < 30:
                return f"🟢 {name}"
            elif age < 120:
                return f"🟡 {name}"
            return f"🔴 {name}"
        
        st.markdown(sensor_status("Seneye", latest_sensors.get('seneye_time')))
        st.markdown(sensor_status("Display (K1)", latest_sensors.get('k1_time')))
        st.markdown(sensor_status("Sump (K2)", latest_sensors.get('k2_time')))
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # ======================== MAIN CONTENT ========================
    
    # Header - compact with reduced spacing
    st.markdown("""
        <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        h1 {margin-bottom: 0rem !important;}
        h3 {margin-top: 0.5rem !important; margin-bottom: 0.5rem !important;}
        hr {margin-top: 0.5rem !important; margin-bottom: 0.5rem !important;}
        </style>
    """, unsafe_allow_html=True)
    
    col_title, col_status = st.columns([4, 1])
    with col_title:
        st.markdown("# 🌊 Nature-Based Treatment Digital Twin")
        st.caption("Girona Pilot • Real-time monitoring • AI-assisted analysis")
    with col_status:
        if not sensor_err:
            st.success("🟢 ONLINE")
        else:
            st.warning("🟡 LIMITED")
    
    st.markdown("---")
    
    # ============ ROW 1: Key Metrics ============
    st.markdown("### 📊 System Performance")
    
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    
    with m1:
        val = latest_sensors.get('seneye_ph')
        st.metric("pH", f"{val:.2f}" if val else "—", delta="sensor")
    with m2:
        val = latest_sensors.get('seneye_temp')
        st.metric("Temp", f"{val:.1f}°C" if val else "—", delta="sensor")
    with m3:
        val = latest_sensors.get('total_ammonia')  # Show calculated total, not free
        st.metric("NH₃ Total", f"{val:.2f}" if val else "—", delta="mg/L")
    with m4:
        val = latest_sensors.get('k1_orp')
        st.metric("ORP", f"{val:.0f} mV" if val else "—", delta="sensor")
    with m5:
        val = latest_pred.get('calc_nh4_removal_pct') if latest_pred else None
        st.metric("NH₄ Removal", f"{val:.0f}%" if val else "—", delta="model")
    with m6:
        val = latest_pred.get('calc_TN_removal_pct') if latest_pred else None
        st.metric("TN Removal", f"{val:.0f}%" if val else "—", delta="model")
    with m7:
        val = latest_pred.get('sim_algae_pond1') if latest_pred else None
        st.metric("Algae", f"{val:.0f} g/m³" if val else "—", delta="model")
    with m8:
        val = latest_pred.get('sim_muALGAE_pond1') if latest_pred else None
        st.metric("Growth", f"{val:.2f} d⁻¹" if val else "—", delta="model")
    
    st.markdown("---")
    
    # ============ ROW 2: Sensor Charts + Chat ============
    left_col, right_col = st.columns([1.3, 0.7])
    
    with left_col:
        st.markdown("### 📈 Live Sensor Data (7 days)")
        
        # Row 1: pH, Temperature, Ammonia, ORP (2x2)
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
        
        with row1_col1:
            # Custom pH plot - Seneye only (Display stuck at 14, Sump unreliable)
            fig_ph = go.Figure()
            if not seneye_recent.empty and 'ph' in seneye_recent.columns:
                data = seneye_recent[seneye_recent['ph'].notna()]
                if not data.empty:
                    fig_ph.add_trace(go.Scatter(
                        x=data['real_timestamp'],
                        y=data['ph'] + PH_OFFSET,
                        mode='lines',
                        name='Seneye',
                        line=dict(color='#1f77b4', width=2)
                    ))
            fig_ph.update_layout(
                title="pH",
                height=220,
                margin=dict(l=10, r=10, t=35, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="",
                yaxis_title=""
            )
            st.plotly_chart(fig_ph, use_container_width=True)
        
        with row1_col2:
            fig = create_multi_sensor_plot(seneye_recent, kactoily1_recent, kactoily2_recent, 'temperature_c', 'Temperature', '(°C)')
            st.plotly_chart(fig, use_container_width=True)
        
        with row2_col1:
            # Show both Free and Total ammonia on same plot
            if not seneye_recent.empty and 'ammonia_mg_l' in seneye_recent.columns:
                fig = go.Figure()
                
                # Free ammonia (Seneye reading)
                data = seneye_recent[seneye_recent['ammonia_mg_l'].notna()]
                if not data.empty:
                    fig.add_trace(go.Scatter(
                        x=data['real_timestamp'],
                        y=data['ammonia_mg_l'],
                        mode='lines',
                        name='Free NH₃',
                        line=dict(color='#ff7f0e', width=2)
                    ))
                
                # Total ammonia equivalent
                data_total = seneye_recent[seneye_recent['total_ammonia'].notna()]
                if not data_total.empty:
                    fig.add_trace(go.Scatter(
                        x=data_total['real_timestamp'],
                        y=data_total['total_ammonia'],
                        mode='lines',
                        name='Total NH₃',
                        line=dict(color='#d62728', width=2)
                    ))
                
                fig.update_layout(
                    title='Ammonia (mg/L)',
                    height=220,
                    margin=dict(l=10, r=10, t=35, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis_title="",
                    yaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ammonia data")
        
        with row2_col2:
            fig = create_multi_sensor_plot(seneye_recent, kactoily1_recent, kactoily2_recent, 'orp', 'ORP', '(mV)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: Salinity, PUR, Conductivity, TDS (2x2)
        st.markdown("### 📡 Additional Sensors")
        row3_col1, row3_col2 = st.columns(2)
        row4_col1, row4_col2 = st.columns(2)
        
        with row3_col1:
            fig = create_multi_sensor_plot(seneye_recent, kactoily1_recent, kactoily2_recent, 'salinity', 'Salinity', '(ppt)')
            st.plotly_chart(fig, use_container_width=True)
        
        with row3_col2:
            fig = create_single_sensor_plot(seneye_recent, 'pur', 'PUR (Seneye)', '#9467bd')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No PUR data")
        
        with row4_col1:
            fig = create_multi_sensor_plot(seneye_recent, kactoily1_recent, kactoily2_recent, 'electrical_conductivity', 'Conductivity', '(mS/cm)')
            st.plotly_chart(fig, use_container_width=True)
        
        with row4_col2:
            fig = create_multi_sensor_plot(seneye_recent, kactoily1_recent, kactoily2_recent, 'specific_gravity', 'Specific Gravity', '')
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 3: Lab Data Plots
        st.markdown("### 🧪 Lab Data History")
        row5_col1, row5_col2 = st.columns(2)
        row6_col1, row6_col2 = st.columns(2)
        row7_col1, row7_col2 = st.columns(2)
        
        with row5_col1:
            fig = create_lab_plot(lab_df, 'Calcium (ppm)', 'Calcium (ppm)', '#e377c2')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Ca data")
        
        with row5_col2:
            fig = create_lab_plot(lab_df, 'Magnesium (ppm)', 'Magnesium (ppm)', '#17becf')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Mg data")
        
        with row6_col1:
            fig = create_lab_plot(lab_df, 'Alkalinity (dKH)', 'Alkalinity (dKH)', '#bcbd22')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Alk data")
        
        with row6_col2:
            fig = create_lab_plot(lab_df, 'Nitrate (ppm)', 'Nitrate (ppm)', '#d62728')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No NO₃ data")
        
        with row7_col1:
            fig = create_lab_plot(lab_df, 'Phosphate (ppm)', 'Phosphate (ppm)', '#9467bd')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No PO₄ data")
        
        with row7_col2:
            fig = create_lab_plot(lab_df, 'pH', 'pH (Lab)', '#1f77b4')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No pH data")
    
    with right_col:
        st.markdown("### 💬 AI Assistant")
        st.caption("Ask questions or say 'run the model'")
        
        chat_container = st.container(height=600)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        question = st.chat_input("Ask anything...")
        
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            ctx = prepare_data_context(latest_sensors, latest_pred, latest_lab, predictions_df)
            response, action_type, action_param = chat_with_claude(question, ctx, st.session_state.chat_history[:-1])
            
            if action_type == 'model':
                # Run model with progress updates
                st.session_state.chat_history.append({"role": "assistant", "content": "🔄 **Running SUMO model...**\n\n`[░░░░░░░░░░░░░░░░░░░░]` 0% - Starting..."})
                success, pred, _ = run_sumo_model()
                if success:
                    st.session_state.chat_history[-1] = {"role": "assistant", "content": format_model_results(pred)}
                else:
                    st.session_state.chat_history[-1] = {"role": "assistant", "content": "❌ Model failed"}
            elif action_type == 'sensitivity':
                # Run sensitivity analysis
                st.session_state.chat_history.append({"role": "assistant", "content": f"📊 **Running sensitivity analysis on {action_param.upper()}...**\n\nThis runs 5 scenarios (-30%, -15%, baseline, +15%, +30%). Please wait..."})
                success, sens_data, _ = run_sensitivity_analysis(action_param)
                if success:
                    st.session_state.sensitivity_results = sens_data  # Store for plotting
                    st.session_state.chat_history[-1] = {"role": "assistant", "content": format_sensitivity_results(sens_data)}
                else:
                    st.session_state.chat_history[-1] = {"role": "assistant", "content": f"❌ Sensitivity analysis failed. Make sure reef_sensitivity_runner.py is in the scripts folder."}
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
        
        # Quick actions
        st.markdown("**Quick actions:**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("▶️ Run", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "Run the model"})
                success, pred, _ = run_sumo_model()
                if success:
                    st.session_state.chat_history.append({"role": "assistant", "content": format_model_results(pred)})
                st.rerun()
        with c2:
            if st.button("📊 Status", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "System status?"})
                ctx = prepare_data_context(latest_sensors, latest_pred, latest_lab, predictions_df)
                r, _, _ = chat_with_claude("Brief system status", ctx, [])
                st.session_state.chat_history.append({"role": "assistant", "content": r})
                st.rerun()
        with c3:
            if st.button("🔬 Explain", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "What limits performance?"})
                ctx = prepare_data_context(latest_sensors, latest_pred, latest_lab, predictions_df)
                r, _, _ = chat_with_claude("What's limiting treatment?", ctx, [])
                st.session_state.chat_history.append({"role": "assistant", "content": r})
                st.rerun()
        with c4:
            if st.button("📈 Sensitivity", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": "Run sensitivity analysis on light"})
                st.session_state.chat_history.append({"role": "assistant", "content": "📊 **Running sensitivity analysis on LIGHT...**\n\nRunning 5 scenarios..."})
                success, sens_data, _ = run_sensitivity_analysis('light')
                if success:
                    st.session_state.sensitivity_results = sens_data
                    st.session_state.chat_history[-1] = {"role": "assistant", "content": format_sensitivity_results(sens_data)}
                st.rerun()
        
        # Model Analysis Section
        st.markdown("---")
        st.markdown("### 🔬 Model Analysis")
        
        if latest_pred:
            # Limiting Factors Bar Chart
            factors = {
                'Light': latest_pred.get('sim_light_avail_pond1', 0) * 100,
                'CO₂': latest_pred.get('sim_CO2_avail_pond1', 0) * 100,
                'NH₄': latest_pred.get('sim_NH4_avail_pond1', 0) * 100,
                'PO₄': latest_pred.get('sim_PO4_avail_pond1', 0) * 100
            }
            
            fig = go.Figure()
            colors = ['#2ca02c' if v > 80 else '#ffbf00' if v > 50 else '#d62728' for v in factors.values()]
            fig.add_trace(go.Bar(
                x=list(factors.keys()),
                y=list(factors.values()),
                marker_color=colors,
                text=[f"{v:.0f}%" for v in factors.values()],
                textposition='outside'
            ))
            fig.update_layout(
                title="Limiting Factors (100% = not limiting)",
                height=220,
                yaxis=dict(range=[0, 110]),
                margin=dict(l=10, r=10, t=35, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            limiting = min(factors, key=factors.get)
            st.warning(f"⚠️ **{limiting}** is most limiting at {factors[limiting]:.0f}%")
            
            # Algae Metrics
            st.markdown("#### 🌿 Algae Performance")
            
            acol1, acol2 = st.columns(2)
            
            with acol1:
                # Algae Biomass Comparison (Pond 1 vs Pond 2)
                algae_data = {
                    'Pond 1': latest_pred.get('sim_algae_pond1', 0),
                    'Pond 2': latest_pred.get('sim_algae_pond2', 0)
                }
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(algae_data.keys()),
                    y=list(algae_data.values()),
                    marker_color=['#2ca02c', '#98df8a'],
                    text=[f"{v:.1f}" for v in algae_data.values()],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Algae Biomass (g COD/m³)",
                    height=200,
                    margin=dict(l=10, r=10, t=35, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with acol2:
                # Growth Rate Comparison
                growth_data = {
                    'Pond 1': latest_pred.get('sim_muALGAE_pond1', 0),
                    'Pond 2': latest_pred.get('sim_muALGAE_pond2', 0)
                }
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(growth_data.keys()),
                    y=list(growth_data.values()),
                    marker_color=['#1f77b4', '#aec7e8'],
                    text=[f"{v:.2f}" for v in growth_data.values()],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Growth Rate (d⁻¹)",
                    height=200,
                    margin=dict(l=10, r=10, t=35, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Nitrogen Removal Performance
            st.markdown("#### 🧪 Nitrogen Removal")
            
            ncol1, ncol2 = st.columns(2)
            
            with ncol1:
                # NH4 through system
                nh4_data = {
                    'Influent': latest_pred.get('model_TKN_gm3', 34.4),
                    'Pond 1': latest_pred.get('sim_nh4_pond1', 0),
                    'Effluent': latest_pred.get('sim_nh4_effluent', 0)
                }
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=list(nh4_data.keys()),
                    y=list(nh4_data.values()),
                    marker_color=['#d62728', '#ff9896', '#2ca02c'],
                    text=[f"{v:.1f}" for v in nh4_data.values()],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="NH₄ Through System (g N/m³)",
                    height=200,
                    margin=dict(l=10, r=10, t=35, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with ncol2:
                # Removal efficiency gauge
                nh4_removal = latest_pred.get('calc_nh4_removal_pct', 0)
                tn_removal = latest_pred.get('calc_TN_removal_pct', 0)
                
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=nh4_removal,
                    title={'text': "NH₄ Removal %"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#2ca02c"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ffcccb"},
                            {'range': [50, 75], 'color': "#ffffcc"},
                            {'range': [75, 100], 'color': "#ccffcc"}
                        ]
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run model to see analysis")
    
    # ============ FULL-WIDTH SECTION: Time Series & Analytics ============
    st.markdown("---")
    st.markdown("### 📉 Model Time Series & Analytics")
    
    ts_data, ts_err = load_timeseries()
    
    if ts_data and 'series' in ts_data:
        st.caption(f"Simulation: {ts_data.get('timestamp', 'Unknown')[:19]} • 100-day projection")
        
        # Row 1: 4 time series plots
        ts_col1, ts_col2, ts_col3, ts_col4 = st.columns(4)
        
        with ts_col1:
            fig = create_timeseries_plot(ts_data, 'algae_pond1', 'Algae Biomass', '#2ca02c', 'algae_pond2', '#98df8a')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with ts_col2:
            fig = create_timeseries_plot(ts_data, 'nh4_removal_pct', 'NH₄ Removal %', '#1f77b4')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with ts_col3:
            fig = create_timeseries_plot(ts_data, 'ph_pond1', 'pH', '#ff7f0e', 'ph_pond2', '#ffbb78')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with ts_col4:
            fig = create_timeseries_plot(ts_data, 'do_pond1', 'Dissolved Oxygen', '#9467bd', 'do_pond2', '#c5b0d5')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: 2 time series + 2 analytical plots
        ts_col5, ts_col6, anal_col1, anal_col2 = st.columns(4)
        
        with ts_col5:
            fig = create_timeseries_plot(ts_data, 'growth_rate_pond1', 'Growth Rate (d⁻¹)', '#17becf', 'growth_rate_pond2', '#9edae5')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with ts_col6:
            fig = create_timeseries_plot(ts_data, 'nh4_pond1', 'NH₄ Concentration', '#d62728', 'nh4_effluent', '#ff9896')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with anal_col1:
            fig = create_sankey_diagram(latest_pred)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with anal_col2:
            fig = create_radar_chart(latest_pred, latest_sensors)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🔄 Run SUMO Model to generate time series and analytics")
    
    # ============ SENSITIVITY ANALYSIS SECTION ============
    if st.session_state.sensitivity_results:
        sens_data = st.session_state.sensitivity_results
        st.markdown("---")
        st.markdown(f"### 📊 Sensitivity Analysis: {sens_data.get('parameter', 'Unknown').upper()}")
        st.caption("Multiple scenarios showing impact of parameter changes on system performance")
        
        # Row of sensitivity plots
        sens_col1, sens_col2, sens_col3, sens_col4 = st.columns(4)
        
        with sens_col1:
            fig = create_sensitivity_timeseries_plot(sens_data, 'algae_pond1', 'Algae Biomass by Scenario')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with sens_col2:
            fig = create_sensitivity_timeseries_plot(sens_data, 'nh4_removal_pct', 'NH₄ Removal % by Scenario')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with sens_col3:
            fig = create_sensitivity_timeseries_plot(sens_data, 'ph_pond1', 'pH by Scenario')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with sens_col4:
            fig = create_sensitivity_timeseries_plot(sens_data, 'growth_rate_pond1', 'Growth Rate by Scenario')
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # INTERACTIVE ANALYTICS SECTION
    # =========================================================================
    
    st.markdown("---")
    st.markdown("## 📊 Interactive Analytics")
    st.caption("Explore correlations, compare time periods, and optimize dosing")
    
    # Create tabs for the three analytics tools
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
        "🔍 Correlation Explorer", 
        "⏰ Time Machine", 
        "💊 Dosing Optimizer"
    ])
    
    # =========================================================================
    # TAB 1: CORRELATION EXPLORER
    # =========================================================================
    with analytics_tab1:
        st.markdown("#### Discover relationships between parameters")
        
        # Build combined dataframe for correlation analysis
        corr_df = pd.DataFrame()
        
        # Add sensor data if available
        if not seneye_recent.empty:
            sensor_corr = seneye_recent[['real_timestamp']].copy()
            sensor_corr['date'] = seneye_recent['real_timestamp'].dt.date
            
            # Add available sensor columns with friendly names
            if 'ph' in seneye_recent.columns:
                sensor_corr['pH'] = seneye_recent['ph'] + PH_OFFSET
            if 'temperature_c' in seneye_recent.columns:
                sensor_corr['Temperature'] = seneye_recent['temperature_c']
            if 'ammonia_mg_l' in seneye_recent.columns:
                sensor_corr['Ammonia'] = seneye_recent['ammonia_mg_l']
            if 'total_ammonia' in seneye_recent.columns:
                sensor_corr['Total Ammonia'] = seneye_recent['total_ammonia']
            
            corr_df = sensor_corr.copy()
        
        # Parameter selection
        if not corr_df.empty and len(corr_df.columns) > 2:
            numeric_cols = [col for col in corr_df.columns if col not in ['real_timestamp', 'date'] and corr_df[col].dtype in ['float64', 'int64']]
            
            if len(numeric_cols) >= 2:
                corr_col1, corr_col2, corr_col3 = st.columns([1, 1, 1])
                
                with corr_col1:
                    x_param = st.selectbox("X-Axis", numeric_cols, index=0, key="corr_x_v6")
                with corr_col2:
                    y_idx = min(1, len(numeric_cols)-1)
                    y_param = st.selectbox("Y-Axis", numeric_cols, index=y_idx, key="corr_y_v6")
                with corr_col3:
                    show_trendline = st.checkbox("Show trendline", value=True, key="corr_trend_v6")
                
                # Filter valid data
                plot_data = corr_df[[x_param, y_param]].dropna()
                
                if len(plot_data) > 5:
                    # Calculate correlation
                    correlation = plot_data[x_param].corr(plot_data[y_param])
                    r_squared = correlation ** 2 if not pd.isna(correlation) else 0
                    
                    # R² indicator
                    if r_squared > 0.49:
                        r2_badge = f"🟢 Strong (R² = {r_squared:.3f})"
                    elif r_squared > 0.16:
                        r2_badge = f"🟡 Moderate (R² = {r_squared:.3f})"
                    else:
                        r2_badge = f"⚪ Weak (R² = {r_squared:.3f})"
                    
                    st.markdown(f"**Correlation:** {r2_badge}")
                    
                    # Create scatter plot
                    fig_corr = px.scatter(
                        plot_data,
                        x=x_param,
                        y=y_param,
                        trendline="ols" if show_trendline else None,
                        opacity=0.6
                    )
                    fig_corr.update_traces(marker=dict(size=8, color='#3b82f6'))
                    fig_corr.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis_title=x_param,
                        yaxis_title=y_param
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Not enough data points for correlation analysis")
            else:
                st.info("Need at least 2 numeric parameters for correlation")
        else:
            st.info("Sensor data not available for correlation analysis")
    
    # =========================================================================
    # TAB 2: TIME MACHINE
    # =========================================================================
    with analytics_tab2:
        st.markdown("#### Compare different time periods")
        
        if not seneye_recent.empty and 'real_timestamp' in seneye_recent.columns:
            # Get date range
            min_date = seneye_recent['real_timestamp'].min().date()
            max_date = seneye_recent['real_timestamp'].max().date()
            
            tm_col1, tm_col2 = st.columns(2)
            
            with tm_col1:
                st.markdown("**🔵 Period A**")
                period_a_start = st.date_input(
                    "Start A", 
                    value=max_date - timedelta(days=5),
                    min_value=min_date,
                    max_value=max_date,
                    key="tm_a_start_v6"
                )
                period_a_end = st.date_input(
                    "End A",
                    value=max_date - timedelta(days=3),
                    min_value=min_date,
                    max_value=max_date,
                    key="tm_a_end_v6"
                )
            
            with tm_col2:
                st.markdown("**🟠 Period B**")
                period_b_start = st.date_input(
                    "Start B",
                    value=max_date - timedelta(days=2),
                    min_value=min_date,
                    max_value=max_date,
                    key="tm_b_start_v6"
                )
                period_b_end = st.date_input(
                    "End B",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="tm_b_end_v6"
                )
            
            # Parameter selection
            tm_params = {'pH': 'ph', 'Temperature': 'temperature_c', 'Ammonia': 'ammonia_mg_l'}
            available_params = {k: v for k, v in tm_params.items() if v in seneye_recent.columns}
            
            if available_params:
                tm_param_label = st.radio(
                    "Parameter to compare",
                    list(available_params.keys()),
                    horizontal=True,
                    key="tm_param_v6"
                )
                tm_param = available_params[tm_param_label]
                
                # Filter data
                period_a_data = seneye_recent[
                    (seneye_recent['real_timestamp'].dt.date >= period_a_start) &
                    (seneye_recent['real_timestamp'].dt.date <= period_a_end)
                ].copy()
                
                period_b_data = seneye_recent[
                    (seneye_recent['real_timestamp'].dt.date >= period_b_start) &
                    (seneye_recent['real_timestamp'].dt.date <= period_b_end)
                ].copy()
                
                if len(period_a_data) > 0 and len(period_b_data) > 0:
                    # Normalize index for overlay
                    period_a_data['idx'] = range(len(period_a_data))
                    period_b_data['idx'] = range(len(period_b_data))
                    
                    # Apply pH offset if needed
                    if tm_param == 'ph':
                        period_a_data['plot_val'] = period_a_data[tm_param] + PH_OFFSET
                        period_b_data['plot_val'] = period_b_data[tm_param] + PH_OFFSET
                    else:
                        period_a_data['plot_val'] = period_a_data[tm_param]
                        period_b_data['plot_val'] = period_b_data[tm_param]
                    
                    # Create overlay plot
                    fig_tm = go.Figure()
                    
                    fig_tm.add_trace(go.Scatter(
                        x=period_a_data['idx'],
                        y=period_a_data['plot_val'],
                        mode='lines',
                        name=f'Period A ({period_a_start} to {period_a_end})',
                        line=dict(color='#3b82f6', width=2)
                    ))
                    
                    fig_tm.add_trace(go.Scatter(
                        x=period_b_data['idx'],
                        y=period_b_data['plot_val'],
                        mode='lines',
                        name=f'Period B ({period_b_start} to {period_b_end})',
                        line=dict(color='#f97316', width=2, dash='dash')
                    ))
                    
                    fig_tm.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis_title="Relative Time (samples)",
                        yaxis_title=tm_param_label,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig_tm, use_container_width=True)
                    
                    # Statistics comparison
                    avg_a = period_a_data['plot_val'].mean()
                    avg_b = period_b_data['plot_val'].mean()
                    std_a = period_a_data['plot_val'].std()
                    std_b = period_b_data['plot_val'].std()
                    delta = avg_b - avg_a
                    delta_pct = (delta / avg_a) * 100 if avg_a != 0 else 0
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    stat_col1.metric("Period A Avg", f"{avg_a:.2f}")
                    stat_col2.metric("Period A Std", f"±{std_a:.3f}")
                    stat_col3.metric("Period B Avg", f"{avg_b:.2f}")
                    stat_col4.metric("Change", f"{delta:+.3f}", delta=f"{delta_pct:+.1f}%")
                else:
                    st.warning("Select date ranges with available data")
        else:
            st.info("Sensor data not available for time comparison")
    
    # =========================================================================
    # TAB 3: DOSING OPTIMIZER WITH SCENARIO FAN
    # =========================================================================
    with analytics_tab3:
        st.markdown("#### Predict alkalinity trends under different dosing scenarios")
        
        if lab_df is not None and len(lab_df) > 0:
            # Get current values
            current_ca = float(latest_lab.get('calcium', 445))
            current_alk = float(latest_lab.get('alkalinity', 9.0))
            current_mg = float(latest_lab.get('magnesium', 1440))
            
            # Targets
            target_ca = 450
            target_alk = 9.2
            target_mg = 1440
            
            # =================== CIRCULAR GAUGES ===================
            def create_gauge_v6(value, target, min_val, max_val, title, unit):
                diff_pct = abs(value - target) / target
                if diff_pct < 0.02:
                    color = "#22c55e"
                    status = "On Target"
                elif diff_pct < 0.05:
                    color = "#eab308"
                    status = "Monitor"
                else:
                    color = "#ef4444"
                    status = "Action Needed"
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=value,
                    delta={'reference': target, 'relative': False, 'valueformat': '.1f', 'position': 'bottom'},
                    number={'suffix': f" {unit}", 'font': {'size': 22}},
                    title={'text': f"<b>{title}</b><br><span style='font-size:11px;color:gray'>{status}</span>"},
                    domain={'x': [0.1, 0.9], 'y': [0.1, 0.9]},
                    gauge={
                        'axis': {'range': [min_val, max_val], 'tickwidth': 1},
                        'bar': {'color': color, 'thickness': 0.7},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#e5e7eb",
                        'steps': [
                            {'range': [min_val, target * 0.92], 'color': '#fee2e2'},
                            {'range': [target * 0.92, target * 0.98], 'color': '#fef9c3'},
                            {'range': [target * 0.98, target * 1.02], 'color': '#dcfce7'},
                            {'range': [target * 1.02, target * 1.08], 'color': '#fef9c3'},
                            {'range': [target * 1.08, max_val], 'color': '#fee2e2'},
                        ],
                        'threshold': {
                            'line': {'color': "#1f2937", 'width': 3},
                            'thickness': 0.8,
                            'value': target
                        }
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=30, r=30, t=50, b=20))
                return fig
            
            # Display gauges
            gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
            
            with gauge_col1:
                st.plotly_chart(create_gauge_v6(current_ca, target_ca, 380, 520, "Calcium", "ppm"), use_container_width=True)
            with gauge_col2:
                st.plotly_chart(create_gauge_v6(current_alk, target_alk, 7.0, 11.0, "Alkalinity", "dKH"), use_container_width=True)
            with gauge_col3:
                st.plotly_chart(create_gauge_v6(current_mg, target_mg, 1300, 1550, "Magnesium", "ppm"), use_container_width=True)
            
            # =================== DOSE SLIDERS ===================
            st.markdown("**Adjust Daily Doses (Red Sea ml/day)**")
            dose_col1, dose_col2, dose_col3 = st.columns(3)
            
            with dose_col1:
                dose_ca = st.slider("Ca+", 0.0, 5.0, 1.7, 0.1, key="dose_ca_v6")
            with dose_col2:
                dose_alk = st.slider("Alk", 0.0, 10.0, 4.2, 0.1, key="dose_alk_v6")
            with dose_col3:
                dose_mg = st.slider("Mg", 0.0, 2.0, 0.2, 0.1, key="dose_mg_v6")
            
            # =================== SCENARIO FAN CHART ===================
            st.markdown("---")
            st.markdown("### 🔮 Scenario Projections")
            st.caption("Historical data → NOW → Three possible futures based on dosing choices")
            
            # Calculate consumption from lab history
            if len(lab_df) >= 3 and 'Date' in lab_df.columns and 'Alkalinity (dKH)' in lab_df.columns:
                recent_lab_df = lab_df.tail(7)
                try:
                    days_span = (pd.to_datetime(recent_lab_df['Date'].iloc[-1]) - pd.to_datetime(recent_lab_df['Date'].iloc[0])).days
                    if days_span > 0:
                        alk_consumption = (recent_lab_df['Alkalinity (dKH)'].iloc[0] - recent_lab_df['Alkalinity (dKH)'].iloc[-1]) / days_span
                        alk_consumption = max(0.01, min(0.2, alk_consumption))
                    else:
                        alk_consumption = 0.05
                except:
                    alk_consumption = 0.05
            else:
                alk_consumption = 0.05
            
            # Effect of dosing: ~0.1 dKH per ml for 38 gal system
            alk_per_ml = 0.1
            
            # Build historical data from lab
            historical_days = 14
            projection_days = 14
            
            historical_data = []
            if 'Date' in lab_df.columns and 'Alkalinity (dKH)' in lab_df.columns:
                recent = lab_df.tail(historical_days).copy()
                recent['Date'] = pd.to_datetime(recent['Date'])
                if len(recent) > 0:
                    max_date_lab = recent['Date'].max()
                    for _, row in recent.iterrows():
                        day_offset = (row['Date'] - max_date_lab).days
                        if pd.notna(row['Alkalinity (dKH)']):
                            historical_data.append({
                                'day': day_offset,
                                'alk': row['Alkalinity (dKH)']
                            })
            
            # If no historical data, create simulated
            if len(historical_data) < 2:
                for d in range(-14, 0):
                    historical_data.append({
                        'day': d,
                        'alk': current_alk + (d * 0.02) + (np.random.random() - 0.5) * 0.1
                    })
            
            # Define scenarios
            scenarios = {
                'optimistic': {'name': 'Increase +25%', 'mult': 1.25, 'color': '#22c55e', 'fill': 'rgba(34, 197, 94, 0.15)'},
                'current': {'name': 'Current Dose', 'mult': 1.0, 'color': '#3b82f6', 'fill': 'rgba(59, 130, 246, 0.15)'},
                'risk': {'name': 'Reduce -25%', 'mult': 0.75, 'color': '#ef4444', 'fill': 'rgba(239, 68, 68, 0.15)'}
            }
            
            # Calculate projections
            for key, scenario in scenarios.items():
                adjusted_dose = dose_alk * scenario['mult']
                daily_change = (adjusted_dose * alk_per_ml) - alk_consumption
                
                scenario['projection'] = []
                scenario['upper'] = []
                scenario['lower'] = []
                
                projected_alk = current_alk
                
                for day in range(projection_days + 1):
                    uncertainty = 0.08 * np.sqrt(day + 1)
                    
                    scenario['projection'].append({'day': day, 'alk': projected_alk})
                    scenario['upper'].append({'day': day, 'alk': projected_alk + uncertainty})
                    scenario['lower'].append({'day': day, 'alk': projected_alk - uncertainty})
                    
                    projected_alk += daily_change
            
            # Build the fan chart
            fig_fan = go.Figure()
            
            # Add uncertainty bands (back to front)
            for key in ['risk', 'current', 'optimistic']:
                scenario = scenarios[key]
                upper_days = [p['day'] for p in scenario['upper']]
                upper_vals = [p['alk'] for p in scenario['upper']]
                lower_days = [p['day'] for p in scenario['lower']][::-1]
                lower_vals = [p['alk'] for p in scenario['lower']][::-1]
                
                fig_fan.add_trace(go.Scatter(
                    x=upper_days + lower_days,
                    y=upper_vals + lower_vals,
                    fill='toself',
                    fillcolor=scenario['fill'],
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add historical line
            hist_days = [p['day'] for p in historical_data]
            hist_vals = [p['alk'] for p in historical_data]
            
            fig_fan.add_trace(go.Scatter(
                x=hist_days,
                y=hist_vals,
                mode='lines+markers',
                name='Historical',
                line=dict(color='#1f2937', width=3),
                marker=dict(size=6)
            ))
            
            # Add projection lines
            for key in ['optimistic', 'current', 'risk']:
                scenario = scenarios[key]
                proj_days = [p['day'] for p in scenario['projection']]
                proj_vals = [p['alk'] for p in scenario['projection']]
                
                fig_fan.add_trace(go.Scatter(
                    x=proj_days,
                    y=proj_vals,
                    mode='lines',
                    name=scenario['name'],
                    line=dict(
                        color=scenario['color'],
                        width=3,
                        dash='solid' if key == 'current' else 'dot'
                    )
                ))
            
            # Add reference lines
            fig_fan.add_vline(x=0, line_width=2, line_dash="dash", line_color="#6b7280",
                             annotation_text="NOW", annotation_position="top")
            fig_fan.add_hline(y=target_alk, line_width=2, line_dash="dash", line_color="#22c55e",
                             annotation_text=f"Target ({target_alk})", annotation_position="right")
            fig_fan.add_hline(y=8.0, line_width=2, line_dash="dash", line_color="#ef4444",
                             annotation_text="Critical (8.0)", annotation_position="right")
            
            # Shade past region
            fig_fan.add_vrect(x0=-historical_days, x1=0, fillcolor="rgba(0,0,0,0.03)", layer="below", line_width=0)
            
            fig_fan.update_layout(
                height=400,
                margin=dict(l=20, r=100, t=30, b=50),
                xaxis_title="Days",
                yaxis_title="Alkalinity (dKH)",
                yaxis=dict(range=[7.0, 11.0]),
                xaxis=dict(range=[-historical_days, projection_days]),
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_fan, use_container_width=True)
            
            # =================== RECOMMENDATION ===================
            final_current = scenarios['current']['projection'][-1]['alk']
            final_optimistic = scenarios['optimistic']['projection'][-1]['alk']
            final_risk = scenarios['risk']['projection'][-1]['alk']
            
            if final_current < 8.3:
                rec_icon, rec_text, rec_color = "🚨", f"**Critical:** Alk will drop to {final_current:.1f} dKH. Increase dose to {dose_alk * 1.3:.1f} ml/day", "#fee2e2"
            elif final_current < target_alk - 0.3:
                rec_icon, rec_text, rec_color = "⚠️", f"**Trending low.** Consider increasing to {dose_alk * 1.15:.1f} ml/day", "#fef3c7"
            elif final_current > target_alk + 0.5:
                rec_icon, rec_text, rec_color = "📉", f"**Will overshoot.** Reduce to {dose_alk * 0.85:.1f} ml/day", "#fef3c7"
            else:
                rec_icon, rec_text, rec_color = "✅", f"**Stable:** Current dosing maintains Alk at {final_current:.1f} dKH", "#dcfce7"
            
            st.markdown(f"""
                <div style='background-color: {rec_color}; padding: 15px 20px; border-radius: 10px;'>
                    <span style='font-size: 20px;'>{rec_icon}</span> {rec_text}
                </div>
            """, unsafe_allow_html=True)
            
            # Scenario summary cards
            st.markdown("<br>", unsafe_allow_html=True)
            sum_col1, sum_col2, sum_col3 = st.columns(3)
            
            with sum_col1:
                st.markdown(f"""
                    <div style='background-color: #dcfce7; padding: 12px; border-radius: 8px; text-align: center; border-left: 4px solid #22c55e;'>
                        <div style='font-size: 11px; color: #166534;'>+25% DOSE</div>
                        <div style='font-size: 24px; font-weight: bold; color: #166534;'>{final_optimistic:.1f}</div>
                        <div style='font-size: 10px; color: #166534;'>dKH in 14 days</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with sum_col2:
                st.markdown(f"""
                    <div style='background-color: #dbeafe; padding: 12px; border-radius: 8px; text-align: center; border-left: 4px solid #3b82f6;'>
                        <div style='font-size: 11px; color: #1e40af;'>CURRENT</div>
                        <div style='font-size: 24px; font-weight: bold; color: #1e40af;'>{final_current:.1f}</div>
                        <div style='font-size: 10px; color: #1e40af;'>dKH in 14 days</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with sum_col3:
                st.markdown(f"""
                    <div style='background-color: #fee2e2; padding: 12px; border-radius: 8px; text-align: center; border-left: 4px solid #ef4444;'>
                        <div style='font-size: 11px; color: #991b1b;'>-25% DOSE</div>
                        <div style='font-size: 24px; font-weight: bold; color: #991b1b;'>{final_risk:.1f}</div>
                        <div style='font-size: 10px; color: #991b1b;'>dKH in 14 days</div>
                    </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("Lab data not available for dosing optimization")


if __name__ == "__main__":
    main()
