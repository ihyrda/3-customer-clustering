"""Streamlit application for the customer-clustering project.

Three tabs: Overview and Data, Cluster Analysis, Customer Assignment.

Run from the project root:

    streamlit run app.py
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import final_feats, get_data_summary, load_data, validate_data
from src.logging_config import configure_logging
from src.modeling import assign_customer, train_clustering_workflow
from src.preprocessing import input_ranges, prepare_customer_input

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Customer Segmentation", layout="wide")


# Streamlit reruns this file after every interaction, so data and fitted models
# use the same caching pattern as Projects 1 and 2.
@st.cache_data
def get_data():
    """Load and validate the dataset once per session."""
    df = load_data()
    validate_data(df)
    return df


@st.cache_resource
def get_clustering_results():
    """Fit the candidate and final clustering models once per session."""
    return train_clustering_workflow(get_data())


# Fail once at startup with a controlled message instead of inside a tab.
try:
    df = get_data()
    results = get_clustering_results()
    summary = get_data_summary(df)
except FileNotFoundError:
    logger.exception("Mall customer dataset was not found")
    st.error("The mall customer dataset could not be found.")
    st.stop()
except ValueError as exc:
    logger.exception("Data validation failed")
    st.error(str(exc))
    st.stop()
except Exception:
    logger.exception("Unexpected application failure")
    st.error("The application encountered an unexpected error.")
    st.stop()


st.title("Customer Segmentation")
st.caption(
    "A modular rebuild of the Unsupervised Clustering notebook. Candidate cluster counts are compared before a final K-means model assigns customers to groups."
)

tab_data, tab_clusters, tab_assign = st.tabs(
    ["Overview and Data", "Cluster Analysis", "Customer Assignment"]
)


# --- Tab 1 -----------------------------------------------------------------
# Dataset shape, missingness, descriptive fields, and preview.
with tab_data:
    st.header("Overview and Data")
    st.write(
        "This is unsupervised learning: No known customer segment. K-means proposes groups from numeric distance, where silhouette measures separation, not predictive accuracy."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", f"{summary['rows']:,}")
    col2.metric("Columns", summary["columns"])
    col3.metric("Missing values", summary["missing_values"])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Numeric feature summary")
        st.dataframe(
            df[final_feats].describe().T.round(2),
            width="stretch",
        )

    with col2:
        st.subheader("Gender")
        counts = summary["gender_counts"]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(counts.keys(), counts.values(), color=["#4C72B0", "#DD8452"])
        ax.set_ylabel("Customers")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.caption(
        "Customer_ID is an identifier and excluded; "
    )
    st.caption(
            "Gender is kept for description, but excluded because K-means requires meaningful numeric distance."
        )

    st.subheader("Data preview")
    st.dataframe(df.head(20), width="stretch")


# --- Tab 2 -----------------------------------------------------------------
# Ordered as the work happened: reproduce the notebook, state what changed and
# why, show the evidence for k, then the result.
with tab_clusters:
    st.header("Cluster Analysis")

    final_metrics = results["final_metrics"]
    two_feature_metrics = results["two_feature_metrics"]
    selected_row = final_metrics.loc[
        final_metrics["Clusters"] == results["selected_k"]
    ].iloc[0]
    notebook_row = two_feature_metrics.loc[
        two_feature_metrics["Clusters"] == results["two_feature_k"]
    ].iloc[0]

    # --- 1. the notebook's setup, recomputed ---
    st.subheader("Reproducing the notebook")
    st.write(
        "The notebook clusters on income and spending score in raw units. That setup is rerun here."
    )
    st.dataframe(
        two_feature_metrics.style.format({
            "Inertia": "{:,.2f}",
            "Silhouette": "{:.4f}",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Highest silhouette selects k = {results['two_feature_k']}, which matches the value the notebook reports."
    )

    st.divider()

    # --- 2. what changed, and the cost of changing it ---
    st.subheader("Adding Age")
    st.write(
        "The final model adds Age and standardizes, so features with different units do not control the distance calculation. So original notebook and final model differ by feature set and scaling."
    )

    comparison = pd.DataFrame({
        "Configuration": ["Notebook", "Final model"],
        "Features": [
            "Income, Spending",
            "Age, Income, Spending",
        ],
        "Scaling": ["Raw units", "Standardized"],
        "Clusters (k)": [
            int(results["two_feature_k"]),
            int(results["selected_k"]),
        ],
        "Silhouette": [
            notebook_row["Silhouette"],
            selected_row["Silhouette"],
        ],
    })
    st.dataframe(
        comparison.style.format({"Silhouette": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

    st.info(
        "Two of its groups hold customers with almost the same income and spending who are around thirty years apart in age. Without Age that separation cannot be represented, scores between models are an  'apples-to-oranges comparison'."
    )
    

    st.divider()

    # --- 3. evidence for k on the final model ---
    st.subheader("Selecting k for the final model")
    st.dataframe(
        final_metrics.style.format({
            "Inertia": "{:,.2f}",
            "Silhouette": "{:.4f}",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Selection rule: highest silhouette score. Inertia elbow view always decreases as clusters are added, so it does not select k but suppors whether higher k added meaningful separation."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(final_metrics["Clusters"], final_metrics["Inertia"], marker="o")
        ax.axvline(results["selected_k"], color="#C44E52", linestyle="--")
        ax.set_xlabel("Clusters (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Final model elbow curve")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            final_metrics["Clusters"],
            final_metrics["Silhouette"],
            marker="o",
        )
        ax.axvline(results["selected_k"], color="#C44E52", linestyle="--")
        ax.set_xlabel("Clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_title("Final model silhouette curve")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.divider()

    # --- 4. the fitted result ---
    st.subheader("Fitted customer clusters")
    st.caption(
        "The same six groups on two and three axes. Only two features fit on the left, so groups separated mainly by Age sit on top of each other there and pull apart on the right."
    )
    assignments = results["assignments"]
    clusters = sorted(assignments["Cluster"].unique())

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        for cluster in clusters:
            group = assignments[assignments["Cluster"] == cluster]
            ax.scatter(
                group["Annual_Income"],
                group["Spending_Score"],
                s=35,
                alpha=0.75,
                label=f"Cluster {cluster}",
            )
        ax.set_xlabel("Annual income (thousands)")
        ax.set_ylabel("Spending score")
        ax.set_title("Income and spending")
        ax.legend(fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col2:
        #3d render for 3 features.
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        for cluster in clusters:
            group = assignments[assignments["Cluster"] == cluster]
            ax.scatter(
                group["Annual_Income"],
                group["Spending_Score"],
                group["Age"],
                s=28,
                alpha=0.85,
                depthshade=False,
                label=f"Cluster {cluster}",
            )
        ax.set_xlabel("Income (thousands)", fontsize=8)
        ax.set_ylabel("Spending score", fontsize=8)
        ax.set_zlabel("Age", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=12, azim=200)
        ax.set_title("Income, spending, and Age")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.subheader("Cluster profiles")
    st.dataframe(
        results["profiles"].style.format({
            "Age": "{:.1f}",
            "Annual_Income": "{:.1f}",
            "Spending_Score": "{:.1f}",
            "Share": "{:.1%}",
        }),
        width="stretch",
        hide_index=True,
    )
    
    st.info(
        "Cluster numbers are labels, interpreted from each group's profile."
    )


# --- Tab 3 -----------------------------------------------------------------
# Three raw inputs enter the same scaled feature space as the fitted centroids.
with tab_assign:
    st.header("Customer Assignment")
    st.write(
        "Enter values within the ranges observed by the model. The fitted scaler transforms the row before K-means assigns its nearest cluster."
    )

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input(
                "Age",
                min_value=input_ranges["Age"][0],
                max_value=input_ranges["Age"][1],
                value=35,
                step=1,
            )
        with col2:
            annual_income = st.number_input(
                "Annual income (thousands)",
                min_value=input_ranges["Annual_Income"][0],
                max_value=input_ranges["Annual_Income"][1],
                value=60,
                step=1,
            )
        with col3:
            spending_score = st.number_input(
                "Spending score",
                min_value=input_ranges["Spending_Score"][0],
                max_value=input_ranges["Spending_Score"][1],
                value=50,
                step=1,
            )

        submitted = st.form_submit_button("Assign cluster")

    if submitted:
        try:
            customer = prepare_customer_input({
                "Age": age,
                "Annual_Income": annual_income,
                "Spending_Score": spending_score,
            })
            cluster = assign_customer(
                results["model"],
                results["scaler"],
                customer,
            )
            profile = results["profiles"].loc[
                results["profiles"]["Cluster"] == cluster
            ].iloc[0]

            st.success(f"Assigned to cluster {cluster}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Average age", f"{profile['Age']:.1f}")
            col2.metric("Average income", f"{profile['Annual_Income']:.1f}k")
            col3.metric("Average spending", f"{profile['Spending_Score']:.1f}")
            st.caption(
                f"This cluster contains {int(profile['Customers'])} customers "
                f"({profile['Share']:.1%} of the dataset)."
            )
        except ValueError as exc:
            logger.exception("Customer input validation failed")
            st.error(str(exc))
        except Exception:
            logger.exception("Unexpected assignment failure")
            st.error("The customer could not be assigned.")

    st.info(
        "Educational demonstration only. These groups describe a small hypothetical dataset and have not been validated against real customer behavior."
    )
