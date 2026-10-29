"""Streamlit dashboard:   streamlit run app.py

Tabs: data overview | models & metrics | prediction | signal timing | fixed vs dynamic
"""
import pandas as pd
import streamlit as st

from src import config as C
from src.data_utils import (clean_data, find_density_column, find_speed_column,
                            load_dataset, overview)
from src.density_model import predict_density, train_density_model
from src.plots import (confusion_heatmap, delay_bars, green_bars, queue_growth, speed_scatter)
from src.signal_optimizer import allocate_green, fixed_plan
from src.simulation import APPROACHES, SCENARIOS, SimConfig, compare_all
from src.speed_model import predict_speed, train_speed_model

st.set_page_config(page_title="Smart Traffic System", page_icon="🚦", layout="wide")
st.title("🚦 AI-Based Smart Traffic Congestion Prediction and Signal Optimization")
st.caption("Phase-I prototype - offline data and a simulated junction. No live cameras or sensors.")

# ------------------------------------------------------------------ data source
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Project dataset / demo", "Upload CSV"])
    uploaded = st.file_uploader("CSV file", type="csv") if source == "Upload CSV" else None


@st.cache_data(show_spinner=False)
def _load(uploaded_bytes, name):
    import io
    up = io.BytesIO(uploaded_bytes) if uploaded_bytes else None
    df, is_demo, msg = load_dataset(uploaded=up)
    df, rep = clean_data(df)
    return df, is_demo, msg, rep


try:
    if source == "Upload CSV" and uploaded is None:
        st.info("Upload a CSV in the sidebar, or switch to 'Project dataset / demo'.")
        st.stop()
    df, is_demo, msg, rep = _load(uploaded.getvalue() if uploaded else None, uploaded.name if uploaded else "")
except ValueError as err:
    st.error(str(err))
    st.stop()

if is_demo:
    st.warning("⚠️ " + msg + "  Prediction numbers on this page are for demonstration only.")
else:
    st.success(msg)

speed_col, dens_col = find_speed_column(df), find_density_column(df)
st.session_state.setdefault("speed", None)
st.session_state.setdefault("density", None)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Data overview", "🤖 Models & metrics", "🔮 Prediction", "🚥 Signal timing", "⚖️ Fixed vs dynamic"])

# ------------------------------------------------------------------ 1. data overview
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (after cleaning)", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicates removed", rep["duplicates_removed"])
    st.write(f"**Speed column found:** `{speed_col}`    |    **Density column found:** `{dens_col}`")
    st.dataframe(df.head(50), use_container_width=True)
    st.subheader("Column summary")
    st.dataframe(overview(df).astype(str), use_container_width=True)

# ------------------------------------------------------------------ 2. models
with tab2:
    if st.button("Train / retrain models", type="primary"):
        with st.spinner("Training (cross-validation on four models)..."):
            try:
                st.session_state["speed"] = train_speed_model(df, speed_col) if speed_col else None
                st.session_state["density"] = train_density_model(df, dens_col) if dens_col else None
            except ValueError as err:
                st.error(str(err))
    if not speed_col:
        st.info("No average-speed column found - speed prediction skipped.")
    if not dens_col:
        st.info("No density/congestion column found - density classification skipped.")
    sp, dn = st.session_state["speed"], st.session_state["density"]
    if sp is None and dn is None:
        st.info("Press **Train / retrain models** to begin.")
    if sp is not None:
        st.subheader("Speed prediction (regression)")
        st.dataframe(sp["results"].round(3), use_container_width=True)
        st.write(f"Best model by cross-validation RMSE: **{sp['best_model']}**")
        st.pyplot(speed_scatter(sp["y_test"], sp["y_pred"]))
    if dn is not None:
        st.subheader("Density classification (Random Forest)")
        st.metric("Test accuracy", f"{dn['accuracy']:.1%}")
        st.dataframe(dn["report"].round(3), use_container_width=True)
        st.dataframe(dn["confusion"], use_container_width=True)
        st.pyplot(confusion_heatmap(dn["confusion"]))
        if is_demo:
            st.caption("Demo labels are built from vehicle_count, so accuracy is higher than real data would give.")

# ------------------------------------------------------------------ 3. prediction
with tab3:
    sp, dn = st.session_state["speed"], st.session_state["density"]
    bundle = (sp or dn or {}).get("bundle")
    if bundle is None:
        st.info("Train the models first (Models & metrics tab).")
    else:
        st.write("Enter conditions for one road segment:")
        info = bundle["feature_info"]
        values, cols = {}, st.columns(3)
        for i, (name, meta) in enumerate(info.items()):
            with cols[i % 3]:
                if meta["type"] == "num":
                    values[name] = st.number_input(name, value=meta["default"], key=f"in_{name}")
                else:
                    opts = meta["options"] or [meta["default"]]
                    values[name] = st.selectbox(name, opts, index=opts.index(meta["default"])
                                                if meta["default"] in opts else 0, key=f"in_{name}")
        if st.button("Predict"):
            row = pd.DataFrame([values])
            c1, c2 = st.columns(2)
            if sp is not None:
                c1.metric("Predicted average speed", f"{predict_speed(sp['bundle'], row)[0]:.1f} km/h")
            if dn is not None:
                label, proba = predict_density(dn["bundle"], row)
                c2.metric("Predicted traffic density", str(label[0]).upper())
                st.bar_chart(proba.T.rename(columns={0: "probability"}))

# ------------------------------------------------------------------ 4. signal timing
with tab4:
    st.write("Share one signal cycle among four roads in proportion to demand. "
             "**Demand here is entered by you / simulated - the dataset has no per-road counts.**")
    c1, c2, c3, c4 = st.columns(4)
    cycle = c1.number_input("Cycle (s)", 40, 300, C.CYCLE_TIME)
    lost = c2.number_input("Lost time / phase (s)", 0, 20, C.LOST_TIME)
    gmin = c3.number_input("Min green (s)", 0, 100, C.MIN_GREEN)
    gmax = c4.number_input("Max green (s)", 1, 200, C.MAX_GREEN)
    dcols = st.columns(4)
    demand = [dcols[i].number_input(f"{a} demand (veh/h)", 0.0, 5000.0, v, 50.0)
              for i, (a, v) in enumerate(zip(APPROACHES, (900.0, 800.0, 250.0, 250.0)))]
    try:
        dyn = allocate_green(demand, cycle, lost, gmin, gmax)
        fix = fixed_plan(4, cycle, lost, gmin, gmax)
        st.dataframe(pd.DataFrame({"Road": APPROACHES, "Demand (veh/h)": demand,
                                   "Fixed green (s)": fix, "Dynamic green (s)": dyn}), use_container_width=True)
        st.caption(f"Green {sum(dyn)} s + lost {4 * lost} s = {sum(dyn) + 4 * lost} s = cycle {cycle} s ✔")
        st.pyplot(green_bars(list(APPROACHES), fix, dyn))
    except ValueError as err:
        st.error(f"Invalid signal settings: {err}")

# ------------------------------------------------------------------ 5. comparison
with tab5:
    st.warning("All results here are from a SIMULATION with simulated demand and an assumed forecast error. "
               "They are not proof of real-world improvement.")
    c1, c2 = st.columns(2)
    runs = c1.slider("Runs per scenario", 1, 20, 5)
    ferr = c2.slider("Forecast error (±%)", 0, 50, int(C.FORECAST_ERROR * 100)) / 100
    with st.expander("Scenarios (vehicles/hour on N, S, E, W)"):
        for k, v in SCENARIOS.items():
            st.write(f"**{k}** - {v['description']}  `{v['rates']}`")
    if st.button("Run simulation", type="primary"):
        bar = st.progress(0.0)
        table, results = compare_all(runs, SimConfig(forecast_error=ferr),
                                     progress=lambda f, n: bar.progress(f, text=n))
        bar.empty()
        st.session_state["sim"] = (table, results)
    if "sim" in st.session_state:
        table, results = st.session_state["sim"]
        st.dataframe(table.round(1), use_container_width=True)
        a, b = st.columns(2)
        a.pyplot(delay_bars(table))
        pick = b.selectbox("Queue growth for scenario", list(results), index=2)
        b.pyplot(queue_growth(results[pick]))
        st.info("Dynamic timing helps when traffic is uneven or shifting, and gives no gain "
                "(or is slightly worse) when traffic is light or perfectly balanced.")
