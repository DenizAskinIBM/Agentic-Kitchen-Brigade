import streamlit as st
import time
import random
import requests
import os
import pandas as pd
# from dotenv import load_dotenv
# load_dotenv()
WEBHOOK_URL = "https://chat.googleapis.com/v1/spaces/AAQAC8Tntc0/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=LOJj-CkvHrJF4MDo5Fd8hnorhDl2qGkg96siegvM120"
THRESHOLDS = {"ROGERS": 350, "BELL": 300, "TELUS": 50}
PROVIDERS = list(THRESHOLDS.keys())
if "counters" not in st.session_state:
    st.session_state.counters = {p: 1 for p in PROVIDERS}
if "alerts_sent" not in st.session_state:
    st.session_state.alerts_sent = {p: False for p in PROVIDERS}
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=PROVIDERS)
def send_alert(provider, count):
    if not WEBHOOK_URL:
        st.warning("Webhook URL not set.")
        return
    message = {"text": f"ALERT: {provider} crossed threshold! Count: {count}"}
    try:
        response = requests.post(WEBHOOK_URL, json=message)
        if response.status_code == 200:
            st.success(f"ALERT: {provider} crossed threshold! Count: {count}")
            #st.write(message)
        else:
            st.error(f"Failed to send alert: {response.status_code}")
    except Exception as e:
        st.error(f"Exception sending alert: {e}")
st.title(":signal_strength: Alert Simulator Dashboard")
col1, col2 = st.columns(2)
if col1.button(":arrow_forward: Start Counter"):
    st.session_state.running = True
if col2.button(":black_square_for_stop: Stop Counter"):
    st.session_state.running = False
metrics_placeholder = st.empty()
chart_placeholder = st.empty()
if st.session_state.running:
    new_row = {
        provider: st.session_state.counters[provider]
        for provider in PROVIDERS
    }
    with metrics_placeholder.container():
        st.subheader(":arrows_counterclockwise: Live Counters")
        for provider in PROVIDERS:
            # change = random.randint(-10, 15)
            # st.session_state.counters[provider] = max(0, st.session_state.counters[provider] + change)
            change = random.randint(1,15)
            st.session_state.counters[provider] = st.session_state.counters[provider] + change
            current_count = st.session_state.counters[provider]
            st.metric(label=provider, value=current_count, delta=change)
            new_row[provider] = current_count
            if current_count >= THRESHOLDS[provider] and not st.session_state.alerts_sent[provider]:
                send_alert(provider, current_count)
                st.session_state.alerts_sent[provider] = True
    # st.session_state.history = pd.concat([
    #     st.session_state.history,
    #     pd.DataFrame([new_row])
    # ], ignore_index=True)
    # chart_placeholder.line_chart(st.session_state.history)
    time.sleep(random.uniform(0.5, 1.5))
    st.rerun()