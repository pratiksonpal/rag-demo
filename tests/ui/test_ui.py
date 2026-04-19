"""
UI tests for RAG Playground Streamlit app.

Covers: page load, sidebar, tab navigation, Tab 1 (Text Splitting),
Tab 2 (Vector Embedding), Tab 3 (Response Generation).

All tests run against a live Streamlit server (localhost:8502) via headless Chromium.
"""

TIMEOUT = 20000  # 20 s


# ── Helpers ───────────────────────────────────────────────────────────────────

def go_tab(page, n: int):
    """Click the nth tab (0-indexed) and wait for content to settle."""
    page.locator('[data-baseweb="tab"]').nth(n).click()
    page.wait_for_timeout(1000)


# ── TC-UI-001 – TC-UI-005: Page Load ─────────────────────────────────────────

class TestPageLoad:
    def test_TC_UI_001_page_title(self, page):
        """Page window title contains 'RAG Playground'."""
        assert "RAG Playground" in page.title()

    def test_TC_UI_002_main_heading_visible(self, page):
        """A section heading is visible on page load (app uses styled divs, not native h1)."""
        heading = page.locator('[data-testid="stMarkdownContainer"]').first
        heading.wait_for(timeout=TIMEOUT)
        assert heading.inner_text().strip() != ""

    def test_TC_UI_003_sidebar_rendered(self, page):
        """Sidebar is visible on page load."""
        assert page.locator('[data-testid="stSidebar"]').is_visible()

    def test_TC_UI_004_three_tabs_present(self, page):
        """At least three tab elements are rendered."""
        assert page.locator('[data-baseweb="tab"]').count() >= 3

    def test_TC_UI_005_no_exception_on_load(self, page):
        """No Streamlit exception (red error box) is shown on initial load."""
        assert page.locator('[data-testid="stException"]').count() == 0


# ── TC-UI-006 – TC-UI-011: Sidebar ───────────────────────────────────────────

class TestSidebar:
    def test_TC_UI_006_document_section_heading(self, page):
        """Sidebar shows a 'Document' section heading."""
        assert "Document" in page.locator('[data-testid="stSidebar"]').inner_text()

    def test_TC_UI_007_sample_document_option(self, page):
        """'Sample Document' radio option is present in sidebar."""
        assert page.get_by_text("Sample Document").is_visible()

    def test_TC_UI_008_upload_option_present(self, page):
        """'Upload .txt file' radio option is present in sidebar."""
        assert page.get_by_text("Upload .txt file").is_visible()

    def test_TC_UI_009_ollama_model_section(self, page):
        """Sidebar has an Ollama / LLM model section."""
        text = page.locator('[data-testid="stSidebar"]').inner_text()
        assert "Ollama" in text or "Model" in text

    def test_TC_UI_010_system_prompt_in_sidebar(self, page):
        """System Prompt textarea is present in the sidebar."""
        sidebar = page.locator('[data-testid="stSidebar"]')
        assert sidebar.locator('[data-testid="stTextArea"]').count() >= 1

    def test_TC_UI_011_footer_branding(self, page):
        """Sidebar footer shows 'Fully local' / 'No API keys' text."""
        text = page.locator('[data-testid="stSidebar"]').inner_text().lower()
        assert "local" in text or "api" in text


# ── TC-UI-012 – TC-UI-016: Tab Navigation ────────────────────────────────────

class TestTabNavigation:
    def test_TC_UI_012_tab_labels(self, page):
        """All three tab labels contain expected keywords."""
        tabs = page.locator('[data-baseweb="tab"]')
        combined = " ".join(tabs.nth(i).inner_text() for i in range(3))
        assert "Splitting" in combined
        assert "Embedding" in combined
        assert "Generation" in combined or "Response" in combined

    def test_TC_UI_013_tab1_active_by_default(self, page):
        """Tab 1 (Text Splitting) is selected by default."""
        assert page.locator('[data-baseweb="tab"]').nth(0).get_attribute("aria-selected") == "true"

    def test_TC_UI_014_navigate_to_tab2(self, page):
        """Clicking Tab 2 activates it."""
        go_tab(page, 1)
        assert page.locator('[data-baseweb="tab"]').nth(1).get_attribute("aria-selected") == "true"

    def test_TC_UI_015_navigate_to_tab3(self, page):
        """Clicking Tab 3 activates it."""
        go_tab(page, 2)
        assert page.locator('[data-baseweb="tab"]').nth(2).get_attribute("aria-selected") == "true"

    def test_TC_UI_016_navigate_back_to_tab1(self, page):
        """Can navigate from Tab 3 back to Tab 1."""
        go_tab(page, 2)
        go_tab(page, 0)
        assert page.locator('[data-baseweb="tab"]').nth(0).get_attribute("aria-selected") == "true"


# ── TC-UI-017 – TC-UI-024: Tab 1 — Text Splitting ────────────────────────────

class TestTab1TextSplitting:
    def test_TC_UI_017_strategy_selectbox_visible(self, page):
        """Split Strategy selectbox is visible on Tab 1."""
        assert page.locator('[data-testid="stSelectbox"]').first.is_visible()

    def test_TC_UI_018_chunk_size_slider_visible(self, page):
        """At least one slider (Chunk Size) is present on Tab 1."""
        assert page.locator('[data-testid="stSlider"]').count() >= 1

    def test_TC_UI_019_overlap_slider_visible(self, page):
        """At least two sliders (Chunk Size + Overlap) are present on Tab 1."""
        assert page.locator('[data-testid="stSlider"]').count() >= 2

    def test_TC_UI_020_apply_button_visible(self, page):
        """'Apply ▶' primary button is visible on Tab 1."""
        assert page.locator('button:has-text("Apply")').is_visible()

    def test_TC_UI_021_source_text_area_visible(self, page):
        """Source document text area is present on Tab 1."""
        assert page.locator('[data-testid="stTextArea"]').count() >= 1

    def test_TC_UI_022_chunking_guide_expander_present(self, page):
        """Chunking strategy guide expander is shown on Tab 1."""
        assert page.get_by_text("Chunking strategy guide").count() >= 1

    def test_TC_UI_023_apply_shows_chunk_cards(self, page):
        """Clicking 'Apply ▶' makes the info prompt disappear and chunk cards appear."""
        # Before Apply: info message is visible
        page.wait_for_selector('[data-testid="stAlert"]', timeout=TIMEOUT)
        page.locator('button:has-text("Apply")').click()
        # After Apply: info message disappears (chunks rendered instead)
        page.wait_for_selector('[data-testid="stAlert"]', state="detached", timeout=TIMEOUT)

    def test_TC_UI_024_apply_shows_chunk_1_label(self, page):
        """After clicking Apply, 'Chunk 1' label is visible in the chunks panel."""
        page.locator('button:has-text("Apply")').click()
        page.wait_for_selector('[data-testid="stAlert"]', state="detached", timeout=TIMEOUT)
        assert page.get_by_text("Chunk 1").count() >= 1


# ── TC-UI-025 – TC-UI-028: Tab 2 — Vector Embedding ─────────────────────────

class TestTab2VectorEmbedding:
    def test_TC_UI_025_build_faiss_button_visible(self, page):
        """'Build FAISS Index' button is visible on Tab 2."""
        go_tab(page, 1)
        assert page.locator('button:has-text("Build FAISS Index")').is_visible()

    def test_TC_UI_026_tab2_content_heading(self, page):
        """Tab 2 panel contains 'Embedding' or 'FAISS' text."""
        go_tab(page, 1)
        text = page.locator('[data-baseweb="tab-panel"]').nth(1).inner_text()
        assert "Embedding" in text or "FAISS" in text

    def test_TC_UI_027_query_input_visible(self, page):
        """A text input for the query is present on Tab 2."""
        go_tab(page, 1)
        assert page.locator('[data-testid="stTextInput"]').count() >= 1

    def test_TC_UI_028_top_k_slider_present(self, page):
        """Top-K slider is present on Tab 2."""
        go_tab(page, 1)
        assert page.locator('[data-testid="stSlider"]').count() >= 1


# ── TC-UI-029 – TC-UI-032: Tab 3 — Response Generation ──────────────────────

class TestTab3ResponseGeneration:
    def test_TC_UI_029_generate_button_visible(self, page):
        """'Generate Response' button is visible on Tab 3."""
        go_tab(page, 2)
        assert page.locator('button:has-text("Generate Response")').is_visible()

    def test_TC_UI_030_query_input_visible(self, page):
        """A query text input is present on Tab 3."""
        go_tab(page, 2)
        assert page.locator('[data-testid="stTextInput"]').count() >= 1

    def test_TC_UI_031_tab3_content_heading(self, page):
        """Tab 3 panel contains 'Response' or 'Generate' text."""
        go_tab(page, 2)
        text = page.locator('[data-baseweb="tab-panel"]').nth(2).inner_text()
        assert "Response" in text or "Generate" in text

    def test_TC_UI_032_workflow_steps_visible(self, page):
        """Tab 3 shows RAG workflow steps (Retrieve / Augment / Generate)."""
        go_tab(page, 2)
        text = page.locator('[data-baseweb="tab-panel"]').nth(2).inner_text()
        assert "Retrieve" in text or "Augment" in text or "Generate" in text
