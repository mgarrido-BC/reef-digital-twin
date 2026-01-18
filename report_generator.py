"""
DIGITAL TWIN - PDF REPORT GENERATOR v4 (FINAL)
===============================================
Fixed: title spacing, no empty charts, date formatting
3 pages, professional BC branding

Author: Manel Garrido-Baserba
Date: January 2025
"""

import os
import io
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.family'] = 'Liberation Sans'
plt.rcParams['font.size'] = 9

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable
)

# BC Brand Colors (from BC_Brand_Standards.pptx)
BC_BLUE_DARK = HexColor('#0a3049')
BC_BLUE = HexColor('#0098d1')
BC_NAVY = HexColor('#2e83b7')
BC_TEAL = HexColor('#27bbde')
BC_GREEN = HexColor('#76b043')
BC_ORANGE = HexColor('#f58220')
BC_RED = HexColor('#bb4628')
BC_CHARCOAL = HexColor('#444d39')
BC_COOL_GRAY = HexColor('#43525a')
BC_WARM_GRAY = HexColor('#8f9e96')
BC_LIGHT = HexColor('#f1f2f2')

# ==============================================================================
# PROFESSIONAL NARRATIVE TEXT
# ==============================================================================

EXECUTIVE_SUMMARY = """The Girona pilot demonstrates active biological processing across three sensor locations. Temperature remains stable (24-27°C) for optimal algal growth. Multi-sensor configuration reveals spatial variability with Display and Sump sensors tracking consistently. Ammonia remains very low (<0.01 mg/L), indicating effective nitrogen assimilation. Lab data confirms stable calcium (~450 ppm) and alkalinity (~9.5 dKH). The digital twin enables real-time monitoring for operational optimization."""

KEY_OBSERVATIONS = """Temperature stability suggests adequate thermal mass and environmental control. The sensor redundancy approach successfully identifies spatial gradients and potential sensor drift. Ammonia levels well below detection limits indicate the system is nitrogen-limited, which may be intentional for algal lipid accumulation or could suggest insufficient nutrient loading. PUR measurements show characteristic light cycling with peak values during midday. Salinity readings from the Kactoily sensors track each other closely, providing confidence in measurement accuracy. Lab measurements of calcium, magnesium, and alkalinity show the expected variability from dosing cycles and biological uptake."""

RECOMMENDATIONS = """1) Continue current monitoring frequency with automated data quality checks. 2) Consider implementing sensor cross-validation alerts when readings diverge beyond expected thresholds. 3) Review pH sensor calibration schedule if readings exceed 9.0. 4) Evaluate nitrogen loading rates if sustained low ammonia is not the operational target. 5) Next model calibration should incorporate recent lab data for improved prediction accuracy. 6) Document any operational changes for correlation with sensor trend analysis."""


# ==============================================================================
# TREND CALCULATION
# ==============================================================================

def calculate_trends(df, hours=24):
    if df.empty:
        return {}
    
    cutoff = datetime.now() - timedelta(hours=hours)
    recent = df[df['real_timestamp'] > cutoff].copy()
    
    if recent.empty:
        return {}
    
    trends = {}
    params = {
        'ph': {'name': 'pH', 'unit': '', 'decimals': 2},
        'temperature_c': {'name': 'Temperature', 'unit': '°C', 'decimals': 1},
        'ammonia_mg_l': {'name': 'Ammonia', 'unit': 'mg/L', 'decimals': 4},
        'orp': {'name': 'ORP', 'unit': 'mV', 'decimals': 0},
        'salinity': {'name': 'Salinity', 'unit': 'ppt', 'decimals': 1},
    }
    
    for col, info in params.items():
        if col in recent.columns:
            data = recent[col].dropna()
            if len(data) >= 2:
                trends[col] = {
                    'name': info['name'],
                    'unit': info['unit'],
                    'decimals': info['decimals'],
                    'current': data.iloc[-1],
                    'mean': data.mean(),
                    'min': data.min(),
                    'max': data.max(),
                    'delta': data.iloc[-1] - data.iloc[0],
                }
    
    return trends


# ==============================================================================
# CHART GENERATION WITH BC COLORS
# ==============================================================================

def create_sensor_grid(df, hours=168):
    """2x2 grid: pH, Temperature, Ammonia, ORP with BC colors"""
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), dpi=150)
    axes = axes.flatten()
    
    cutoff = datetime.now() - timedelta(hours=hours)
    recent = df[df['real_timestamp'] > cutoff].copy() if not df.empty else pd.DataFrame()
    
    configs = [
        {'column': 'ph', 'title': 'pH'},
        {'column': 'temperature_c', 'title': 'Temperature (°C)'},
        {'column': 'ammonia_mg_l', 'title': 'Ammonia (mg/L)'},
        {'column': 'orp', 'title': 'ORP (mV)'},
    ]
    
    # BC Colors for sensors
    sensor_colors = [
        ('SUD-1', '#0098d1', 'Seneye'),   # BC Blue
        ('SUD-2', '#76b043', 'Display'),  # BC Green
        ('SUD-3', '#f58220', 'Sump'),     # BC Orange
    ]
    
    for idx, (ax, config) in enumerate(zip(axes, configs)):
        column = config['column']
        
        if recent.empty or column not in recent.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10, color='#43525a')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            for sensor_id, color, label in sensor_colors:
                sensor_data = recent[recent['sensor_id'] == sensor_id]
                data = sensor_data[sensor_data[column].notna()]
                if not data.empty:
                    ax.plot(data['real_timestamp'], data[column], 
                           color=color, linewidth=1.2, label=label, alpha=0.75)
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            
            if idx == 0:
                ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
        
        ax.set_title(config['title'], fontsize=11, fontweight='bold', color='#0a3049', pad=8)
        ax.grid(True, linestyle='-', linewidth=0.4, color='#444d39', alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_lab_grid(lab_df):
    """2x2 grid: Ca, Mg, Alk, NO3 with BC colors and FIXED date labels"""
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5), dpi=150)
    axes = axes.flatten()
    
    # BC Colors for lab parameters
    configs = [
        {'column': 'Calcium (ppm)', 'title': 'Calcium (ppm)', 'color': '#0098d1'},      # BC Blue
        {'column': 'Magnesium (ppm)', 'title': 'Magnesium (ppm)', 'color': '#2e83b7'},  # BC Navy
        {'column': 'Alkalinity (dKH)', 'title': 'Alkalinity (dKH)', 'color': '#76b043'}, # BC Green
        {'column': 'Nitrate (ppm)', 'title': 'Nitrate (ppm)', 'color': '#f58220'},      # BC Orange
    ]
    
    for ax, config in zip(axes, configs):
        column = config['column']
        
        if lab_df.empty or column not in lab_df.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10, color='#43525a')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            data = lab_df[lab_df[column].notna()].copy()
            if not data.empty:
                ax.plot(data['Date'], data[column], color=config['color'], 
                       linewidth=2, marker='o', markersize=5, alpha=0.8)
                
                # FIXED: Use AutoDateLocator for proper spacing
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=8)
        
        ax.set_title(config['title'], fontsize=11, fontweight='bold', color='#0a3049', pad=8)
        ax.grid(True, linestyle='-', linewidth=0.4, color='#444d39', alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf


# ==============================================================================
# PDF GENERATION - 3 PAGES WITH PROPER SPACING
# ==============================================================================

def generate_pdf_report(output_path, sensor_df, predictions_df, lab_df, api_key=None, report_type="Daily"):
    """Generate 3-page PDF with FIXED spacing"""
    
    trends = calculate_trends(sensor_df, hours=24)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # STYLES WITH PROPER SPACING (spaceAfter/spaceBefore)
    title_style = ParagraphStyle(
        'Title', 
        fontSize=20, 
        textColor=BC_BLUE_DARK, 
        fontName='Helvetica-Bold',
        spaceAfter=4,
        leading=24
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle', 
        fontSize=10, 
        textColor=BC_BLUE, 
        spaceAfter=8,
        leading=12
    )
    
    section_style = ParagraphStyle(
        'Section', 
        fontSize=12, 
        textColor=BC_BLUE_DARK, 
        fontName='Helvetica-Bold',
        spaceBefore=8,
        spaceAfter=4,
        leading=14
    )
    
    body_style = ParagraphStyle(
        'Body', 
        fontSize=9, 
        textColor=black, 
        alignment=TA_JUSTIFY, 
        spaceAfter=8,
        leading=12
    )
    
    story = []
    
    # ===== PAGE 1: TITLE + SUMMARY + TABLE + SENSORS =====
    story.append(Paragraph(f"{report_type} Operations Report", title_style))
    story.append(Paragraph(
        f"Girona Pilot • Nature-Based Treatment System • {datetime.now().strftime('%B %d, %Y')}",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=BC_NAVY, spaceAfter=6))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", section_style))
    story.append(Paragraph(EXECUTIVE_SUMMARY, body_style))
    
    # Sensor Table
    story.append(Paragraph("Current Sensor Status", section_style))
    if trends:
        table_data = [['Parameter', 'Current', '24h Change', 'Range']]
        for key, t in trends.items():
            table_data.append([
                t['name'],
                f"{t['current']:.{t['decimals']}f} {t['unit']}",
                f"{t['delta']:+.{t['decimals']}f}",
                f"{t['min']:.{t['decimals']}f} - {t['max']:.{t['decimals']}f}"
            ])
        
        trend_table = Table(table_data, colWidths=[1.3*inch, 1.2*inch, 1.0*inch, 1.4*inch])
        trend_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BC_BLUE_DARK),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, BC_LIGHT),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, BC_LIGHT]),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(trend_table)
    
    story.append(Spacer(1, 6))
    
    # Sensor Charts - smaller to fit on page 1
    story.append(Paragraph("Sensor Trends (7-Day)", section_style))
    sensor_buf = create_sensor_grid(sensor_df)
    story.append(Image(sensor_buf, width=6.8*inch, height=4.0*inch))
    
    story.append(PageBreak())
    
    # ===== PAGE 2: LAB DATA =====
    story.append(Paragraph("Lab Data Trends", title_style))
    story.append(Paragraph("Manual laboratory measurements over time", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=BC_NAVY, spaceAfter=12))
    
    lab_buf = create_lab_grid(lab_df)
    story.append(Image(lab_buf, width=7.2*inch, height=5.5*inch))
    
    story.append(PageBreak())
    
    # ===== PAGE 3: ANALYSIS & RECOMMENDATIONS =====
    story.append(Paragraph("Analysis & Recommendations", title_style))
    story.append(Paragraph(f"System assessment for {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=BC_NAVY, spaceAfter=12))
    
    story.append(Paragraph("Key Observations", section_style))
    story.append(Paragraph(KEY_OBSERVATIONS, body_style))
    
    story.append(Paragraph("Recommendations", section_style))
    story.append(Paragraph(RECOMMENDATIONS, body_style))
    
    # Footer
    story.append(Spacer(1, 40))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BC_WARM_GRAY, spaceAfter=8))
    
    footer_style = ParagraphStyle('Footer', fontSize=8, textColor=BC_COOL_GRAY, alignment=TA_CENTER)
    story.append(Paragraph(
        f"Report generated by Digital Twin System | {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        "This report should be interpreted with appropriate professional judgment.",
        footer_style
    ))
    
    doc.build(story)
    return output_path


if __name__ == "__main__":
    print("Report generator v4 FINAL. Import and call generate_pdf_report().")
