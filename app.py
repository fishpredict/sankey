import pandas as pd
import numpy as np
import plotly.graph_objects as go
import colorsys
import random
from dash import Dash, html, dcc, callback_context, Output, Input, State
import dash_bootstrap_components as dbc
from collections import defaultdict
import os
import sys
import json
import psutil


# Local configuration with 500 inferences file
data_path = "./data/gpt2_med_env30_abu100_240ep_final_model_500inf_test_dataset.csv"
species_dict_path = "./data/dict_id2species.txt"


# ==================== MEMORY MONITORING FUNCTIONS ====================
# Simplified functions for faster startup

def get_memory_usage():
    """Returns the current memory usage of the process."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    return memory_mb

def log_memory_usage(step=""):
    """Logs memory usage with a descriptive message."""
    memory_mb = get_memory_usage()
    print(f"Memory usage {step}: {memory_mb:.2f} MB")
    return memory_mb

# ==================== DATA PROCESSING FUNCTIONS ====================

def load_and_preprocess_data(file_path):
    """Load and prepare data for analysis - simplified version for faster startup."""
    print(f"Starting data preprocessing from: {file_path}")
    
    # Load species dictionary
    try:
        print(f"Loading species dictionary from: {species_dict_path}")
        with open(species_dict_path, 'r') as f:
            species_dict = json.load(f)
        print(f"Species dictionary loaded with {len(species_dict)} entries")
    except Exception as e:
        print(f"Dictionary error: {e}")
        species_dict = {}
    
    try:
        print(f"Loading CSV data from: {file_path}")
        
        # Simple and fast loading - without complex memory optimizations
        df = pd.read_csv(file_path)
        
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"Column names: {list(df.columns)[:10]}...")  # First 10 columns
        
    except Exception as e:
        print(f"Data loading error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Identify species columns
    species_cols = [col for col in df.columns if col.startswith('SP')]
    
    # Create a dictionary of species present per sample
    sample_species = {}
    for _, row in df.iterrows():
        sample_key = (row['sample_index'], row['sample_num'])
        species_present = []
        
        for col in species_cols:
            if row[col] > 0:
                species_present.append((col, row[col]))
        
        # Sort by decreasing abundance
        species_present.sort(key=lambda x: x[1], reverse=True)
        
        # Store species sequence
        if species_present:
            sample_species[sample_key] = species_present
    
    # Create summary of most frequent species
    species_freq = defaultdict(int)
    for species_list in sample_species.values():
        for species, _ in species_list:
            species_freq[species] += 1
    
    # Calculate site statistics
    sites = sorted(df['sample_index'].unique())
    
    # Create color map for all species in dataset (not just dictionary)
    all_species = species_cols  # Use actual species columns from dataset
    species_colors = generate_color_palette(len(all_species))
    species_color_map = {sp: color for sp, color in zip(all_species, species_colors)}
    
    result = {
        'df': df,
        'species_cols': species_cols,
        'sample_species': sample_species,
        'species_freq': dict(species_freq),
        'sites': sites,
        'top_species': sorted(species_freq.items(), key=lambda x: x[1], reverse=True),
        'species_dict': species_dict,
        'species_color_map': species_color_map
    }
    
    print(f"Data preprocessing completed successfully")
    print(f"Found {len(species_cols)} species columns")
    print(f"Processed {len(sample_species)} samples")
    print(f"Created color map for {len(species_color_map)} species")
    
    return result

def generate_color_palette(n_colors, saturation=0.75, value=0.95, alpha=0.8):
    """Generate a palette of n distinct colors with improved saturation and contrast."""
    colors = []
    
    # Predefined colors with more saturation but still harmonious
    enhanced_colors = [
        f"rgba(204, 121, 167, {alpha})",    # Rose
        f"rgba(86, 180, 86, {alpha})",      # Medium green
        f"rgba(213, 94, 94, {alpha})",      # Medium red
        f"rgba(86, 180, 180, {alpha})",     # Medium turquoise
        f"rgba(215, 180, 76, {alpha})",     # Medium gold
        f"rgba(120, 120, 204, {alpha})",    # Medium blue
        f"rgba(225, 153, 76, {alpha})",     # Medium orange
        f"rgba(153, 84, 204, {alpha})",     # Medium violet
        f"rgba(86, 153, 204, {alpha})",     # Medium sky blue
        f"rgba(204, 76, 153, {alpha})",     # Medium magenta
        f"rgba(153, 204, 76, {alpha})",     # Medium lime
        f"rgba(229, 153, 153, {alpha})",    # Medium coral
        f"rgba(76, 153, 76, {alpha})",      # Forest green
        f"rgba(153, 76, 76, {alpha})",      # Brick red
        f"rgba(76, 76, 153, {alpha})",      # Navy blue
        f"rgba(172, 115, 57, {alpha})",     # Brown
        f"rgba(204, 204, 57, {alpha})",     # Enhanced yellow
        f"rgba(204, 57, 204, {alpha})",     # Enhanced violet
        f"rgba(57, 204, 204, {alpha})",     # Enhanced cyan
        f"rgba(229, 115, 57, {alpha})",     # Enhanced orange
        f"rgba(120, 57, 204, {alpha})",     # Enhanced indigo
        f"rgba(57, 204, 120, {alpha})",     # Enhanced jade
        f"rgba(204, 57, 115, {alpha})",     # Enhanced rose
        f"rgba(153, 172, 230, {alpha})",    # Soft periwinkle
        f"rgba(230, 153, 172, {alpha})",    # Soft pink
        f"rgba(172, 230, 153, {alpha})",    # Soft mint
        f"rgba(230, 230, 153, {alpha})",    # Soft yellow
        f"rgba(153, 230, 230, {alpha})",    # Soft cyan
        f"rgba(230, 153, 230, {alpha})",    # Soft lavender
        f"rgba(132, 94, 57, {alpha})",      # Sienna
        f"rgba(76, 128, 57, {alpha})",      # Olive green
        f"rgba(57, 76, 128, {alpha})",      # Slate blue
        f"rgba(128, 57, 76, {alpha})",      # Burgundy
        f"rgba(57, 128, 94, {alpha})",      # Dark turquoise
        f"rgba(94, 57, 128, {alpha})",      # Amethyst
        f"rgba(235, 194, 57, {alpha})",     # Amber gold
        f"rgba(57, 172, 172, {alpha})",     # Dark aqua
        f"rgba(172, 57, 102, {alpha})",     # Raspberry
        f"rgba(102, 172, 57, {alpha})",     # Apple green
        f"rgba(235, 91, 172, {alpha})",     # Bright pink
    ]
    
    # Use predefined colors first
    colors.extend(enhanced_colors[:min(len(enhanced_colors), n_colors)])
    
    # If more colors are needed, generate them algorithmically with improved saturation
    if n_colors > len(enhanced_colors):
        remaining = n_colors - len(enhanced_colors)
        # Use the golden ratio to maximize hue difference
        golden_ratio_conjugate = 0.618033988749895
        h = 0.5  # Start with a somewhat random hue
        
        for i in range(remaining):
            # Use the golden ratio to generate distinct hues
            h = (h + golden_ratio_conjugate) % 1.0
            # Use high saturation and value for more vivid colors
            rgb = colorsys.hsv_to_rgb(h, 0.75, 0.95)
            
            # Adjust RGB values to maintain contrast
            r = int(rgb[0] * 215 + 40)  # Base of 40 for more saturation
            g = int(rgb[1] * 215 + 40)
            b = int(rgb[2] * 215 + 40)
            
            # Limit values to valid range
            r = min(r, 255)
            g = min(g, 255)
            b = min(b, 255)
            
            color = f"rgba({r}, {g}, {b}, {alpha})"
            colors.append(color)
            
    return colors

#






 


# ==================== SANKEY DIAGRAM GENERATION FUNCTIONS ====================

def create_all_sequences_sankey(data, filtered_sites=None, filtered_species=None, 
                               max_species=5, first_species_colors=True):
    """
    Create a Sankey diagram for all sequences.
    """
    df = data['df']
    sample_species = data['sample_species']
    species_dict = data['species_dict']
    species_color_map = data['species_color_map']
    
    # Filter by site if necessary
    if filtered_sites:
        filtered_df = df[df['sample_index'].isin(filtered_sites)]
        filtered_keys = [(row['sample_index'], row['sample_num']) for _, row in filtered_df.iterrows()]
        filtered_samples = {k: v for k, v in sample_species.items() if k in filtered_keys}
    else:
        filtered_samples = sample_species
    
    # Prepare data for the diagram
    links = []
    node_colors = {}  # Store colors for each node
    
    # Start node color in gray
    node_colors["Start"] = "rgba(100,100,100,0.8)"
    
    # For each sample
    for (site, rep), species_list in filtered_samples.items():
        # Filter by species if necessary
        if filtered_species:
            species_list = [(sp, val) for sp, val in species_list if sp in filtered_species]
        
        # Limit to max_species
        species_list = species_list[:max_species]
        
        # Create links for this sequence
        if species_list:
            first_species = species_list[0][0]
            first_species_id = first_species
            
            # Retrieve the color of the first species (handle missing keys)
            if first_species_id in species_color_map:
                first_species_color = species_color_map[first_species_id]
            else:
                # Default bright color if not found
                first_species_color = "rgba(255,105,180,0.8)"  # Default bright pink
            
            # Retrieve the full name of the species if available
            species_name = species_dict.get(first_species_id.replace("SP", "SP"), first_species_id)
            
            # Add the node color for the first species (if not already defined)
            node_key = f"R1: {species_name}"
            if node_key not in node_colors:
                node_colors[node_key] = first_species_color
            
            # Link from start to the first species
            links.append({
                "source": "Start", 
                "target": node_key, 
                "value": 1,
                "color": first_species_color  # Use the color of the first species for the link
            })
            
            # Links between consecutive species
            for i in range(len(species_list) - 1):
                source_species = species_list[i][0]
                target_species = species_list[i+1][0]
                
                # Retrieve full names
                source_name = species_dict.get(source_species.replace("SP", "SP"), source_species)
                target_name = species_dict.get(target_species.replace("SP", "SP"), target_species)
                
                source_key = f"R{i+1}: {source_name}"
                target_key = f"R{i+2}: {target_name}"
                
                # Target species color
                if target_species in species_color_map:
                    target_color = species_color_map[target_species]
                else:
                    target_color = "rgba(135,206,250,0.8)"  # Default sky blue
                
                # Add the target node color (if not already defined)
                if target_key not in node_colors:
                    node_colors[target_key] = target_color
                
                links.append({
                    "source": source_key, 
                    "target": target_key, 
                    "value": 1,
                    "color": first_species_color  # Always the color of the first species
                })
    
    # Build the diagram with node colors
    return build_sankey_diagram(links, node_colors, title="")

def create_site_repetitions_sankey(data, selected_sites, max_species=5):
    """
    Create a Sankey diagram for selected sites with 500 colors for repetitions.
    """
    df = data['df']
    sample_species = data['sample_species']
    species_dict = data['species_dict']
    species_color_map = data['species_color_map']
    
    # Filter for selected sites
    filtered_df = df[df['sample_index'].isin(selected_sites)]
    
    # Determine the maximum number of repetitions in the data
    max_repetitions = filtered_df['sample_num'].max() if not filtered_df.empty else 500
    # Ensure at least 500 colors to support all cases
    num_colors = max(500, max_repetitions + 50)  # +50 for safety margin
    
    # Dynamic palette for repetitions
    repetition_colors = generate_color_palette(num_colors, saturation=0.95, value=0.95)
    
    print(f"Generated {num_colors} colors for up to {max_repetitions} repetitions")
    
    # Prepare data for the diagram
    links = []
    node_colors = {}  # Store colors for each node
    
    # For each site, add a site node with a specific color
    for site in selected_sites:
        site_key = f"Survey {site}"
        site_color = "rgba(80,80,80,0.8)"  # Dark gray for site nodes
        node_colors[site_key] = site_color
    
    # For each row in the filtered dataframe
    for _, row in filtered_df.iterrows():
        site = row['sample_index']
        rep = row['sample_num']
        key = (site, rep)
        
        if key in sample_species:
            species_list = sample_species[key][:max_species]
            
            if species_list:
                first_species = species_list[0][0]
                
                # Repetition color
                rep_color = repetition_colors[rep % len(repetition_colors)]
                
                # Species color
                if first_species in species_color_map:
                    species_color = species_color_map[first_species]
                else:
                    species_color = "rgba(255,105,180,0.8)"  # Default color
                
                # Full species name
                species_name = species_dict.get(first_species.replace("SP", "SP"), first_species)
                node_key = f"R1: {species_name}"
                
                # Add the node color for the first species (if not already defined)
                if node_key not in node_colors:
                    node_colors[node_key] = species_color
                
                # Link from site to the first species
                links.append({
                    "source": f"Survey {site}", 
                    "target": node_key, 
                    "value": 1,
                    "color": rep_color  # Use repetition color for links
                })
                
                # Links between consecutive species
                for i in range(len(species_list) - 1):
                    source_species = species_list[i][0]
                    target_species = species_list[i+1][0]
                    
                    # Full names
                    source_name = species_dict.get(source_species.replace("SP", "SP"), source_species)
                    target_name = species_dict.get(target_species.replace("SP", "SP"), target_species)
                    
                    source_key = f"R{i+1}: {source_name}"
                    target_key = f"R{i+2}: {target_name}"
                    
                    # Target species color
                    if target_species in species_color_map:
                        target_color = species_color_map[target_species]
                    else:
                        target_color = "rgba(135,206,250,0.8)"  # Default color
                    
                    # Add the target node color (if not already defined)
                    if target_key not in node_colors:
                        node_colors[target_key] = target_color
                    
                    links.append({
                        "source": source_key, 
                        "target": target_key, 
                        "value": 1,
                        "color": rep_color  # Always the repetition color
                    })
    
    # Build the diagram with node colors
    return build_sankey_diagram(links, node_colors, title="")

def build_sankey_diagram(links, node_colors=None, title=""):
    """Build a Sankey diagram from the provided links."""
    if not links:
        # Return empty diagram if no links
        return go.Figure().update_layout(
            title="No data to display with current filters",
            height=800
        )
    
    # Extract unique nodes
    unique_nodes = set()
    for link in links:
        unique_nodes.add(link["source"])
        unique_nodes.add(link["target"])
    
    node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
    
    # Prepare data for Plotly
    sources = [node_to_idx[link["source"]] for link in links]
    targets = [node_to_idx[link["target"]] for link in links]
    values = [link["value"] for link in links]
    link_colors = [link["color"] for link in links]
    
    # Transform labels - remove "Rx: " prefix for display
    labels = []
    for node in unique_nodes:
        if node == "Start" or node.startswith("Survey"):
            # Keep "Start" and "Survey X" as is
            labels.append(node)
        elif node.startswith("R") and ": " in node:
            # Remove "Rx: " prefix from node labels
            parts = node.split(": ", 1)
            if len(parts) > 1:
                labels.append(parts[1])  # Keep only the species name
            else:
                labels.append(node)  # Fallback if unexpected
        else:
            labels.append(node)
    
    # Prepare node colors with a default bright color
    default_color = "rgba(100,100,100,0.8)"  # Default dark gray
    
    # Create mapping of original nodes to transformed labels index
    original_to_label_idx = {node: i for i, node in enumerate(unique_nodes)}
    
    if node_colors:
        # Create a list of colors corresponding to the order of nodes
        node_color_list = []
        for node in unique_nodes:
            color = node_colors.get(node, default_color)
            node_color_list.append(color)
        
        node_config = dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_color_list
        )
    else:
        # Generate a bright palette if no node color provided
        auto_colors = generate_color_palette(len(labels), saturation=0.95, value=0.95, alpha=0.8)
        node_config = dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=labels,
            color=auto_colors
        )
    
    # Create the diagram
    fig = go.Figure(data=[go.Sankey(
        node=node_config,
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])
    
    # Update layout
    fig.update_layout(
        title_text=title,
        font_size=10,
        height=800,
        dragmode="zoom",
        modebar_add=["zoomIn2d", "zoomOut2d", "resetScale2d", "toImage"],
        modebar=dict(orientation='v'),
        paper_bgcolor='rgba(255,255,255,1)',  # White background
        plot_bgcolor='rgba(255,255,255,1)'    # White background
    )
    
    return fig


# ==================== DASH APPLICATION ====================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Required for Hugging Face Spaces

# Load data
try:
    print(f"Loading data from: {data_path}")
    data = load_and_preprocess_data(data_path)
    if data:
        print(f"{len(data['sites'])} sites, {len(data['species_cols'])} species")
        print(f"Data structure keys: {list(data.keys())}")
    else:
        print("Loading failed - data is None")
        raise Exception("Data loading returned None")
except Exception as e:
    print(f"Critical Error during data loading: {e}")
    import traceback
    traceback.print_exc()
    print("Application cannot continue without data")
    sys.exit(1)

# Layout definition
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Species Sequences - Sankey Diagram", 
                   style={"textAlign": "center", "marginTop": "20px"}),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Visualization Parameters"),
                dbc.CardBody([
                    html.Div(id='site-selection-container', children=[
                        html.H5("Survey to Display"),
                        dcc.Dropdown(
                            id='site-selector',
                            options=[{'label': f'Site {site}', 'value': site} for site in data['sites']],
                            multi=False,
                            placeholder="Select a survey (leave empty for all surveys)",
                            style={"marginBottom": "20px"}
                        )
                    ]),
                    
                    html.H5("Maximum Sequence Length"),
                    dcc.Slider(
                        id='max-species-slider',
                        min=2,
                        max=15,
                        step=1,
                        value=4,
                        marks={i: str(i) for i in range(2, 16)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.H5("Zoom Level", style={"marginTop": "20px"}),
                    dcc.Slider(
                        id='zoom-slider',
                        min=0.5,
                        max=2.0,
                        step=0.1,
                        value=1.0,
                        marks={i/10: str(i/10) for i in range(5, 21, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Button(
                        "Update Diagram", 
                        id='update-button',
                        className="btn btn-success mt-3 w-100",
                        disabled=True  # Disabled by default
                    ),
                    
                    html.Button(
                        "Export as HTML", 
                        id='export-button',
                        className="btn btn-info mt-2 w-100",
                        disabled=True  # Disabled by default
                    )
                ])
            ], className="mb-4")
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sankey Diagram"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-sankey",
                        type="circle",
                        children=[
                            dcc.Graph(
                                id='sankey-graph',
                                figure={},
                                style={'height': '700px'},
                                config={
                                    'scrollZoom': True,
                                    'displayModeBar': True,
                                    'editable': True,
                                    'toImageButtonOptions': {
                                        'format': 'svg', 
                                        'filename': 'sankey_diagram',
                                        'height': 800,
                                        'width': 1100,
                                        'scale': 2
                                    }
                                }
                            )
                        ]
                    )
                ])
            ])
        ], width=9)
    ]),
    
    dcc.Download(id="download-html"),
    
    dbc.Row([
        dbc.Col([
            html.Footer(
                "",
                style={"textAlign": "center", "marginTop": "20px", "marginBottom": "10px"}
            )
        ])
    ])
], fluid=True)


# ==================== DASH CALLBACKS ====================

@app.callback(
    Output('sankey-graph', 'figure'),
    Input('update-button', 'n_clicks'),
    State('site-selector', 'value'),
    State('max-species-slider', 'value'),
    State('zoom-slider', 'value'),
    prevent_initial_call=False
)
def update_sankey(n_clicks, selected_site, max_species, zoom_level):
    """Update the Sankey diagram based on selected parameters."""
    ctx = callback_context
    is_initial = ctx.triggered_id is None
    
        # Prepare filtered sites list
    filtered_sites = [selected_site] if selected_site is not None else None
    
    if is_initial:
        # On initial load, display an instruction message
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="<----- Select a survey to display",
            showarrow=False,
            font=dict(
                size=20,
                color="black"
            )
        )
        fig.update_layout(
            height=700
        )
        return fig
    else:
        # Always colored by the first species
        fig = create_all_sequences_sankey(
            data,
            filtered_sites=filtered_sites,
            filtered_species=None,  # No more species filtering
            max_species=max_species,
            first_species_colors=True
        )
    
    # Apply zoom level
    if zoom_level != 1.0:
        fig.update_layout(
            font_size=10 * zoom_level,
            height=800 * zoom_level
        )
    
    return fig

@app.callback(
    Output("download-html", "data"),
    Input("export-button", "n_clicks"),
    State('site-selector', 'value'),
    State('max-species-slider', 'value'),
    prevent_initial_call=True
)
def export_html(n_clicks, selected_site, max_species):
    """Export the current diagram as interactive HTML."""
    if n_clicks is None:
        return None
    
        # Prepare filtered sites list
    filtered_sites = [selected_site] if selected_site is not None else None
    
    # Always colored by the first species
    fig = create_all_sequences_sankey(
        data,
        filtered_sites=filtered_sites,
        filtered_species=None,  # No more species filtering
        max_species=max_species,
        first_species_colors=True
    )
    
    # Export configuration
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'editable': True,
        'toImageButtonOptions': {
            'format': 'svg', 
            'filename': 'sankey_diagram',
            'height': 800,
            'width': 1100,
            'scale': 2
        }
    }
    
    # Create the HTML
    html_str = fig.to_html(include_plotlyjs=True, full_html=True, config=config)
    
    # Return content for download
    return dict(
        content=html_str,
        filename="sankey_diagram_interactive.html"
    )

@app.callback(
    [Output('update-button', 'disabled'),
     Output('export-button', 'disabled')],
    Input('site-selector', 'value')
)
def toggle_button_state(selected_site):
    """Enable or disable buttons based on site selection."""
    buttons_disabled = True if selected_site is None else False
    return buttons_disabled, buttons_disabled

def export_standalone_html(filename="sankey_visualization.html"):
    """Generate a standalone HTML file with the interactive diagram."""
    print(f"Generating standalone HTML file '{filename}'...")
    
    # Create the figure with current parameters
    fig = create_all_sequences_sankey(
        data,
        max_species=15  # You can adjust parameters if needed
    )
    
    # Full interactive configuration
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'editable': True,
        'toImageButtonOptions': {
            'format': 'svg', 
            'filename': 'sankey_diagram',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }
    
    # Save as HTML with all dependencies embedded
    fig.write_html(
        filename,
        include_plotlyjs=True,  # Embed the entire Plotly library
        full_html=True,         # Create a complete HTML document
        config=config           # Add interactive configuration
    )
    
    print(f"Standalone HTML file successfully generated: {filename}")
    print(f"Location: {os.path.abspath(filename)}")
    
    return os.path.abspath(filename)



# Unified entry point with auto-detection
if __name__ == '__main__':
    print("Application startup initiated...")

    print(f"Application will be available at: http://0.0.0.0:7860")
    
    try:
        app.run_server(debug=True, host='0.0.0.0', port=7860)
    except Exception as e:
        print(f"Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
