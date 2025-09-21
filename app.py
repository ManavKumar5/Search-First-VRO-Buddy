import streamlit as st
from search_api import SearchEngine
import os

st.set_page_config(page_title="VRO Buddy — Search Demo", layout="wide")


@st.cache_resource
def get_engine():
    return SearchEngine()  # class constructor, no args


engine = get_engine()


if "clicks" not in st.session_state:
    st.session_state["clicks"] = {}

st.title("VRO Buddy — Search-First Demo")
st.markdown(
    "Enter a keyword query (e.g. 'blue sneakers under 5,000') and optionally filter by category and price range."
)

with st.sidebar:
    st.header("Filters")
    categories = ["All"] + sorted(list(engine.df["category"].dropna().unique()))
    selected_cat = st.selectbox("Category", categories)
    min_price = st.number_input("Min price", value=0)
    max_price = st.number_input("Max price", value=int(engine.df["price"].max()))


query = st.text_input("Search for products", "")

if st.button("Search") or query:
    # Run search
    results = engine.search(
        query,
        category=None if selected_cat == "All" else selected_cat,
        min_price=min_price,
        max_price=max_price,
    )

    if len(results) == 0:
        st.warning("No products found. Try adjusting your search or filters.")
    else:
        st.subheader(f"Search Results ({len(results)} found)")

        cols = st.columns(3)  # display in 3 columns grid
        for idx, row in enumerate(results):
            with cols[idx % 3]:
                st.image(row["image"], width=150)
                st.markdown(f"**{row['name']}**")
                st.caption(row["description"])
                st.write(f"💰 ₹{row['price']}")
                st.write(f"📦 {row['category']}")

                # Add both row id + loop index to ensure uniqueness
                if st.button(f"Like {row['id']}", key=f"like_{row['id']}_{idx}"):
                    if row["category"] not in st.session_state["clicks"]:
                        st.session_state["clicks"][row["category"]] = 0
                    st.session_state["clicks"][row["category"]] += 1
                    st.success(f"Added preference for {row['category']}")


# Show personalization insights
if st.session_state["clicks"]:
    st.sidebar.subheader("Your Preferences")
    for cat, count in st.session_state["clicks"].items():
        st.sidebar.write(f"{cat}: {count} clicks")
