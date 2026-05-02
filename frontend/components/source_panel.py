import streamlit as st


def render_sources(sources) -> None:
    if not sources:
        return
    st.divider()
    st.subheader("📚 Nguồn tài liệu tham khảo")
    for i, doc in enumerate(sources):
        source_name = doc.metadata.get("title", f"Nguồn tài liệu #{i+1}")
        with st.expander(f"📖 [{i+1}] {source_name}"):
            st.markdown("**Nội dung**")
            st.info(doc.page_content)
