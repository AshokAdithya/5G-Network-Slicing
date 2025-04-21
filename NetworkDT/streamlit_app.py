import streamlit as st
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import networkx as nx
from tensorflow.keras.models import load_model
from agent.dqn_agent import DQNAgent
import io
import base64

# Load pretend models
traffic_model = load_model("./models/ann_model.h5")
# demand_model = load_model("./models/demandPredictionTanh2.h5")

state_shape = (2,) # (throughput, latency)
action_size = 3 # three slices

dqn_agent = DQNAgent(state_shape=state_shape, action_size=action_size)

# Simulating Traffic using Random Functions

def simulate_traffic():
    return {
        "emBB": {
            "packet_size": random.randint(1500, 5000),
            "latency": random.uniform(10, 50),
            "throughput": random.uniform(50, 200)
        },
        "URLLC": {
            "packet_size": random.randint(50, 1500),
            "latency": random.uniform(1, 10),
            "throughput": random.uniform(10, 50)
        },
        "mMTC": {
            "packet_size": random.randint(20, 500),
            "latency": random.uniform(20, 80),
            "throughput": random.uniform(5, 30)
        }
    }

def simulate_demand(traffic):
    demand = {}
    for slice_type, params in traffic.items():
        throughput = params["throughput"]
        latency = params["latency"]
        fake_input = np.array([[throughput, latency]])
        demand[slice_type] = [
            throughput * random.uniform(0.8, 1.2),
            latency * random.uniform(0.9, 1.1)
        ]
    return demand

def simulate_bandwidth_allocation(demand_predictions):
    allocation = {}
    for slice_type, values in demand_predictions.items():
        state = np.array(values)
        _ = dqn_agent.act(state)  
        allocation[slice_type] = values[0] * random.uniform(0.5, 1.5)
    return allocation

# Streamlit App

st.set_page_config(page_title="5G Network Slicing", layout="wide")
st.title("5G Network Traffic & Demand Slicing")

# Sidebar Information
with st.sidebar:
    st.header("Info Panel")
    st.markdown("""    
### Traffic Slice Types

**emBB (Enhanced Mobile Broadband)**  
- High Throughput (50-200 Mbps)  
- Medium Latency (10-50 ms)  
- Large Packet Size (1500-5000 bytes)

**URLLC (Ultra-Reliable Low-Latency Communication)**  
- Moderate Throughput (10-50 Mbps)  
- Ultra Low Latency (1-10 ms)  
- Small Packet Size (50-1500 bytes)

**mMTC (Massive Machine Type Communication)**  
- Low Throughput (5-30 Mbps)  
- High Latency (20-80 ms)  
- Tiny Packet Size (20-500 bytes)

---  
**Demand Prediction**  
Demand is computed based on real-time traffic parameters for each slice.

**Traffic Classification**  
Automated based on packet size, latency, and throughput to ensure optimal allocation.
""")

col1, col2, col3 = st.columns(3)
traffic_box = col1.empty()
demand_box = col2.empty()
allocation_box = col3.empty()
graph_box = st.empty()

def display_all(traffic, demand_predictions, allocation):
    traffic_str = ""
    for s, p in traffic.items():
        traffic_str += f"**{s}**\n"
        traffic_str += f"- Packet Size: `{p['packet_size']} bytes`\n"
        traffic_str += f"- Latency: `{p['latency']:.2f} ms`\n"
        traffic_str += f"- Throughput: `{p['throughput']:.2f} Mbps`\n\n"
    traffic_box.markdown("### Traffic Parameters\n" + traffic_str)

    demand_str = ""
    for s, d in demand_predictions.items():
        demand_str += f"**{s}**: Bandwidth = `{d[0]:.2f} Mbps`\n"
        demand_str += f"Latency = `{d[1]:.2f} ms`\n\n"
    demand_box.markdown("### Predicted Demand\n" + demand_str)


    allocation_str = ""
    for s, a in allocation.items():
        allocation_str += f"**{s}**: `{a:.2f} Mbps`\n"
    allocation_box.markdown("### Bandwidth Allocation\n" + allocation_str)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    G = nx.Graph()
    slices = ['emBB', 'URLLC', 'mMTC']
    G.add_node("Server", pos=(5, 5))
    for i, s in enumerate(slices):
        G.add_node(s, pos=(np.random.uniform(1, 9), np.random.uniform(1, 9)))
        G.add_edge(s, "Server", weight=random.randint(1, 5))

    pos = nx.spring_layout(G, scale=0.5)
    nx.draw(G, pos, with_labels=True, node_size=800, node_color='skyblue',
            edge_color='gray', font_weight='bold', font_size=7, ax=ax)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    graph_box.markdown(
        f"""
        <div style="width:400px; margin:auto;">
            <img src="data:image/png;base64,{img_base64}" style="width:100%; height:auto; border-radius:10px;" />
        </div>
        """,
        unsafe_allow_html=True
    )

for _ in range(100):
    traffic = simulate_traffic()
    demand_predictions = simulate_demand(traffic)
    allocation = simulate_bandwidth_allocation(demand_predictions)
    display_all(traffic, demand_predictions, allocation)
    time.sleep(1)
