"""Color palette utilities for spatial omics visualization."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import colorsys
from typing import List


def generate_plotly_extended_palette(n_colors: int) -> List[tuple]:
    """Generate extended Plotly/D3 color palette with up to 60 distinct colors."""
    # Original 10 Plotly/D3 colors
    base_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf'   # cyan
    ]
    
    palette = []
    
    # Convert base colors to RGB
    for color_hex in base_colors:
        rgb = mcolors.to_rgb(color_hex)
        palette.append(rgb)
    
    # If we need more than 10 colors, generate variations
    if n_colors > 10:
        # Generate lighter variants (colors 11-20)
        for color_hex in base_colors:
            r, g, b = mcolors.to_rgb(color_hex)
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            # Lighter version: increase value, slightly decrease saturation
            v_light = min(1.0, v * 1.3)
            s_light = max(0.3, s * 0.7)
            r_light, g_light, b_light = colorsys.hsv_to_rgb(h, s_light, v_light)
            palette.append((r_light, g_light, b_light))
            if len(palette) >= n_colors:
                break
    
    # If we need more than 20 colors, generate darker variants (colors 21-30)
    if n_colors > 20:
        for color_hex in base_colors:
            r, g, b = mcolors.to_rgb(color_hex)
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            # Darker version: decrease value, increase saturation
            v_dark = max(0.3, v * 0.6)
            s_dark = min(1.0, s * 1.2)
            r_dark, g_dark, b_dark = colorsys.hsv_to_rgb(h, s_dark, v_dark)
            palette.append((r_dark, g_dark, b_dark))
            if len(palette) >= n_colors:
                break
    
    # If we need more than 30 colors, generate desaturated variants (colors 31-40)
    if n_colors > 30:
        for color_hex in base_colors:
            r, g, b = mcolors.to_rgb(color_hex)
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            # Desaturated version: decrease saturation, maintain value
            s_desat = max(0.2, s * 0.4)
            v_desat = min(1.0, v * 1.1)
            r_desat, g_desat, b_desat = colorsys.hsv_to_rgb(h, s_desat, v_desat)
            palette.append((r_desat, g_desat, b_desat))
            if len(palette) >= n_colors:
                break
    
    # If we need more than 40 colors, generate hue-shifted variants (colors 41-50)
    if n_colors > 40:
        for color_hex in base_colors:
            r, g, b = mcolors.to_rgb(color_hex)
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            # Hue-shifted version: shift hue by 30 degrees
            h_shift = (h + 0.083) % 1.0  # 30 degrees = 30/360 = 0.083
            r_shift, g_shift, b_shift = colorsys.hsv_to_rgb(h_shift, s, v)
            palette.append((r_shift, g_shift, b_shift))
            if len(palette) >= n_colors:
                break
    
    # If we need more than 50 colors, generate high-saturation variants (colors 51-60)
    if n_colors > 50:
        for color_hex in base_colors:
            r, g, b = mcolors.to_rgb(color_hex)
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            # High-saturation version: maximize saturation, adjust value
            s_high = 1.0
            v_high = max(0.5, min(1.0, v * 0.9))
            r_high, g_high, b_high = colorsys.hsv_to_rgb(h, s_high, v_high)
            palette.append((r_high, g_high, b_high))
            if len(palette) >= n_colors:
                break
    
    return palette[:n_colors] if n_colors <= 60 else palette


def generate_custom_turbo_palette(n_colors: int) -> List[tuple]:
    """Generate Custom Turbo color palette - vibrant, high-contrast colors."""
    # Use matplotlib's turbo colormap as base
    cmap = plt.get_cmap("turbo", n_colors)
    palette = []
    
    for i in range(n_colors):
        rgb = cmap(i)[:3]  # Get RGB, ignore alpha
        palette.append(rgb)
    
    return palette


def generate_sns_palette(n_colors: int) -> List[tuple]:
    """Generate seaborn color palette with multiple beautiful options."""
    # Use different seaborn palettes in sequence for variety
    palette = []
    
    # Base seaborn palettes to cycle through
    sns_palettes = ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
    
    colors_per_palette = max(1, n_colors // len(sns_palettes) + 1)
    
    for i, palette_name in enumerate(sns_palettes):
        try:
            pal = sns.color_palette(palette_name, colors_per_palette)
            palette.extend(pal)
            if len(palette) >= n_colors:
                break
        except:
            # Fallback to default if palette doesn't exist
            pal = sns.color_palette("husl", colors_per_palette)
            palette.extend(pal)
            if len(palette) >= n_colors:
                break
    
    # If we still need more colors, add husl colors
    if len(palette) < n_colors:
        remaining = n_colors - len(palette)
        extra_colors = sns.color_palette("husl", remaining)
        palette.extend(extra_colors)
    
    return palette[:n_colors]


def generate_milliomics_palette(n_colors: int) -> List[tuple]:
    """Generate Milliomics brand color palette using DD596B, 313131, and 6D9F37."""
    # Milliomics brand colors
    brand_colors = [
        "#DD596B",  # Milliomics pink/red
        "#6D9F37",  # Milliomics green
        "#313131",  # Milliomics dark gray
    ]
    
    palette = []
    
    # Convert brand colors to RGB
    for color_hex in brand_colors:
        rgb = mcolors.to_rgb(color_hex)
        palette.append(rgb)
    
    # If we need more colors, generate variations of the brand colors
    if n_colors > 3:
        # Generate lighter variants
        for color_hex in brand_colors:
            if len(palette) >= n_colors:
                break
            r, g, b = mcolors.to_rgb(color_hex)
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            # Lighter version
            v_light = min(1.0, v * 1.4)
            s_light = max(0.3, s * 0.8)
            r_light, g_light, b_light = colorsys.hsv_to_rgb(h, s_light, v_light)
            palette.append((r_light, g_light, b_light))
        
        # Generate darker variants
        for color_hex in brand_colors:
            if len(palette) >= n_colors:
                break
            r, g, b = mcolors.to_rgb(color_hex)
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            # Darker version
            v_dark = max(0.2, v * 0.6)
            s_dark = min(1.0, s * 1.2)
            r_dark, g_dark, b_dark = colorsys.hsv_to_rgb(h, s_dark, v_dark)
            palette.append((r_dark, g_dark, b_dark))
        
        # Generate desaturated variants
        for color_hex in brand_colors:
            if len(palette) >= n_colors:
                break
            r, g, b = mcolors.to_rgb(color_hex)
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            # Desaturated version
            s_desat = max(0.2, s * 0.5)
            v_desat = min(1.0, v * 1.1)
            r_desat, g_desat, b_desat = colorsys.hsv_to_rgb(h, s_desat, v_desat)
            palette.append((r_desat, g_desat, b_desat))
        
        # If still need more, generate hue-shifted variants
        while len(palette) < n_colors:
            for color_hex in brand_colors:
                if len(palette) >= n_colors:
                    break
                r, g, b = mcolors.to_rgb(color_hex)
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                # Hue-shifted version
                h_shift = (h + 0.15) % 1.0  # 54 degrees shift
                r_shift, g_shift, b_shift = colorsys.hsv_to_rgb(h_shift, s, v)
                palette.append((r_shift, g_shift, b_shift))
    
    return palette[:n_colors] 