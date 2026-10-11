import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from data_pipeline import load_and_preprocess
from attack_engine import inject_label_flipping
from defense_engine import sanitize_data

st.set_page_config(page_title="Enterprise ML Poison Defense", layout="wide")
st.title("🛡️ Enterprise ML: Adversarial Poisoning Detection & Defense")
st.markdown("**Core Security Engine:** Empirical Data-Level Attack Simulation & Pre-Training Quarantine")

@st.cache_data
def get_data():
    return load_and_preprocess()

pipe = get_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Ingestion & Baseline", 
    "2. Attack Simulation & Tampered Data", 
    "3. Pre-Training Quarantine Log", 
    "4. Audit & Verification"
])

# ==========================================
# TAB 1: Ingestion & Baseline
# ==========================================
with tab1:
    st.subheader("Data Ingestion & Clean Training Partition")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Live Sample Records from `creditcard.csv`:")
        st.dataframe(pipe["raw_df"][['V1', 'V2', 'V3', 'Amount', 'Class']].head(6), use_container_width=True)
    
    with col2:
        st.metric("Total Ingested Records", f"{len(pipe['raw_df']):,}")
        st.metric("Training Pool Size", len(pipe["X_train"]))
        st.metric("Holdout Validation Size", len(pipe["X_test"]))
    
    st.markdown("---")
    if st.button("🚀 Train Clean Baseline Model"):
        with st.spinner("Fitting model on clean partitions..."):
            clf = RandomForestClassifier(n_estimators=40, random_state=42)
            clf.fit(pipe["X_train_scaled"], pipe["y_train"])
            preds = clf.predict(pipe["X_test_scaled"])
            
            st.session_state['clean_acc'] = accuracy_score(pipe["y_test"], preds)
            st.session_state['clean_rec'] = recall_score(pipe["y_test"], preds)
            st.session_state['clean_f1'] = f1_score(pipe["y_test"], preds)
            st.session_state['clean_cm'] = confusion_matrix(pipe["y_test"], preds)
            st.session_state['base_done'] = True
            
        st.success(f"Clean Model Trained on Real Telemetry! Accuracy: {st.session_state['clean_acc']*100:.2f}% | Fraud Recall: {st.session_state['clean_rec']*100:.2f}%")

# ==========================================
# TAB 2: Attack Simulation & Real Altered Data
# ==========================================
with tab2:
    st.subheader("Simulate Adversarial Contamination in Training Pool")
    st.markdown("Inject label-inversion poison directly into the active training DataFrame.")
    
    poison_pct = st.slider("Poison Level (% of Fraud samples flipped to Benign)", 5, 50, 25)
    
    if st.button("⚠️ Inject Adversarial Poison"):
        with st.spinner("Injecting poison and retraining model..."):
            y_poisoned, poisoned_idx, count, tampered_df = inject_label_flipping(
                pipe["X_train"], pipe["y_train"], poison_pct / 100.0
            )
            
            clf_p = RandomForestClassifier(n_estimators=40, random_state=42)
            clf_p.fit(pipe["X_train_scaled"], y_poisoned)
            p_preds = clf_p.predict(pipe["X_test_scaled"])
            
            st.session_state['poison_acc'] = accuracy_score(pipe["y_test"], p_preds)
            st.session_state['poison_rec'] = recall_score(pipe["y_test"], p_preds)
            st.session_state['poison_f1'] = f1_score(pipe["y_test"], p_preds)
            st.session_state['poison_cm'] = confusion_matrix(pipe["y_test"], p_preds)
            
            st.session_state['y_poisoned'] = y_poisoned
            st.session_state['poisoned_idx'] = poisoned_idx
            st.session_state['tampered_df'] = tampered_df
            st.session_state['atk_done'] = True
            
        st.error(f"Pipeline Compromised! Fraud Recall dropped to: {st.session_state['poison_rec']*100:.2f}% ({count} records tampered)")

    if 'atk_done' in st.session_state:
        st.markdown("---")
        st.write("🔍 **Empirical Proof: Verified Altered Rows Inside Memory**")
        st.caption("These specific rows were intercepted and their ground-truth labels corrupted from 1 to 0:")
        st.dataframe(
            st.session_state['tampered_df'][['Original_Label', 'Injected_Poison_Label', 'Attack_Status', 'V1', 'V2', 'Amount']].head(10),
            use_container_width=True
        )
        st.download_button(
            "📥 Download Injected Poison Audit Log (CSV)",
            st.session_state['tampered_df'].to_csv().encode('utf-8'),
            "poison_audit_log.csv",
            "text/csv"
        )

# ==========================================
# TAB 3: Pre-Training Quarantine Log
# ==========================================
with tab3:
    st.subheader("Pre-Training Data Sanitization (Isolation Forest)")
    
    if st.button("🛡️ Execute Pre-Training Sanitizer"):
        if 'atk_done' not in st.session_state:
            st.warning("Run Tab 2 (Attack Simulation) first.")
        else:
            with st.spinner("Scanning feature space with Isolation Forest..."):
                X_san, y_san, q_idx, quar_df, n_clean, n_quar = sanitize_data(
                    pipe["X_train"], st.session_state['y_poisoned']
                )
                
                # Check real overlap between injected indices and quarantined indices
                caught = len(q_idx.intersection(st.session_state['poisoned_idx']))
                total_p = len(st.session_state['poisoned_idx'])
                catch_rate = (caught / total_p) * 100 if total_p > 0 else 0
                
                # Tag quarantined rows with whether they were genuine poison
                quar_df["Verified_Poison_Target"] = [idx in st.session_state['poisoned_idx'] for idx in quar_df.index]
                
                X_san_scaled = pipe["scaler"].transform(X_san)
                clf_d = RandomForestClassifier(n_estimators=40, random_state=42)
                clf_d.fit(X_san_scaled, y_san)
                d_preds = clf_d.predict(pipe["X_test_scaled"])
                
                st.session_state['def_acc'] = accuracy_score(pipe["y_test"], d_preds)
                st.session_state['def_rec'] = recall_score(pipe["y_test"], d_preds)
                st.session_state['def_f1'] = f1_score(pipe["y_test"], d_preds)
                st.session_state['def_cm'] = confusion_matrix(pipe["y_test"], d_preds)
                st.session_state['quar_df'] = quar_df
                st.session_state['catch_rate'] = catch_rate
                st.session_state['n_quar'] = n_quar
                st.session_state['X_san'] = X_san
                st.session_state['y_san'] = y_san
                st.session_state['def_done'] = True
                
            st.success("Sanitization completed. Inconsistent records quarantined prior to downstream training.")

    if 'def_done' in st.session_state:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows Sent to Quarantine", st.session_state['n_quar'])
        c2.metric("Poison Caught by Algorithm", f"{st.session_state['catch_rate']:.1f}%")
        c3.metric("Restored Fraud Recall", f"{st.session_state['def_rec']*100:.2f}%")
        
        st.write("📋 **Quarantine Registry (Records Excluded From Training):**")
        st.dataframe(
            st.session_state['quar_df'][['Isolation_Status', 'Verified_Poison_Target', 'Assigned_Label', 'V1', 'V2', 'Amount']].head(10),
            use_container_width=True
        )
        
        # Download sanitized clean training dataset
        sanitized_export = st.session_state['X_san'].copy()
        sanitized_export['Class'] = st.session_state['y_san']
        st.download_button(
            "📥 Download Verified Sanitized CSV",
            sanitized_export.to_csv(index=False).encode('utf-8'),
            "sanitized_training_data.csv",
            "text/csv"
        )

# ==========================================
# TAB 4: Evaluation Benchmark
# ==========================================
with tab4:
    st.subheader("Audit & Comparative Benchmark")
    if 'def_done' in st.session_state:
        df_metrics = pd.DataFrame({
            "Metric": ["Overall Accuracy (%)", "Fraud Recall / Sensitivity (%)", "F1-Score"],
            "1. Clean Baseline": [
                f"{st.session_state['clean_acc']*100:.2f}%", 
                f"{st.session_state['clean_rec']*100:.2f}%", 
                f"{st.session_state['clean_f1']:.4f}"
            ],
            "2. Attacked Pipeline": [
                f"{st.session_state['poison_acc']*100:.2f}%", 
                f"{st.session_state['poison_rec']*100:.2f}%", 
                f"{st.session_state['poison_f1']:.4f}"
            ],
            "3. Defended (Sanitized)": [
                f"{st.session_state['def_acc']*100:.2f}%", 
                f"{st.session_state['def_rec']*100:.2f}%", 
                f"{st.session_state['def_f1']:.4f}"
            ]
        })
        st.table(df_metrics.set_index("Metric"))
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
        sns.heatmap(st.session_state['clean_cm'], annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
        axes[0].set_title("1. Clean Baseline")
        sns.heatmap(st.session_state['poison_cm'], annot=True, fmt='d', cmap='Reds', ax=axes[1], cbar=False)
        axes[1].set_title("2. Poisoned (Attacked)")
        sns.heatmap(st.session_state['def_cm'], annot=True, fmt='d', cmap='Greens', ax=axes[2], cbar=False)
        axes[2].set_title("3. Defended (Sanitized)")
        st.pyplot(fig)
    else:
        st.info("Execute tabs 1, 2, and 3 to generate the comparative audit.")