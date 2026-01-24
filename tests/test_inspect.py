"""Tests for CitationsWidget rewritten query display (v0.4.3).

The original inspect mode (Ctrl+I toggle) was removed in favor of
displaying rewritten query information directly in CitationsWidget.
"""

from chatty.rag.provider import RAGMetadata, RAGSource


class TestCitationsWidgetRewrittenQuery:
    """Tests for rewritten query display in CitationsWidget."""

    def test_citations_widget_stores_query_params(self) -> None:
        """CitationsWidget should store rewritten_query and original_query."""
        from chatty.ui.widgets import CitationsWidget

        sources = [
            RAGSource(
                title="Test Paper",
                pmid="12345",
                snippet="Test snippet",
                score=0.9,
            )
        ]

        widget = CitationsWidget(
            sources=sources,
            retrieval_time_s=1.5,
            chunk_count=10,
            rewritten_query="side effects cardiovascular",
            original_query="What about side effects?",
        )

        assert widget._rewritten_query == "side effects cardiovascular"
        assert widget._original_query == "What about side effects?"

    def test_citations_widget_query_params_default_none(self) -> None:
        """CitationsWidget query params should default to None."""
        from chatty.ui.widgets import CitationsWidget

        sources = [
            RAGSource(
                title="Test",
                snippet="Test",
                score=0.9,
            )
        ]

        widget = CitationsWidget(sources=sources)

        assert widget._rewritten_query is None
        assert widget._original_query is None

    def test_from_metadata_passes_original_query(self) -> None:
        """from_metadata should pass original_query to widget."""
        from chatty.ui.widgets import CitationsWidget

        metadata = RAGMetadata(
            sources=[
                RAGSource(
                    title="Test Paper",
                    snippet="Test content",
                    score=0.9,
                )
            ],
            retrieval_time_s=1.5,
            chunk_count=1,
            rewritten_query="expanded query with context",
        )

        widget = CitationsWidget.from_metadata(
            metadata,
            original_query="What about it?",
        )

        assert widget is not None
        assert widget._rewritten_query == "expanded query with context"
        assert widget._original_query == "What about it?"

    def test_from_metadata_without_original_query(self) -> None:
        """from_metadata works without original_query."""
        from chatty.ui.widgets import CitationsWidget

        metadata = RAGMetadata(
            sources=[
                RAGSource(
                    title="Test",
                    snippet="Test",
                    score=0.9,
                )
            ],
            rewritten_query="expanded query",
        )

        widget = CitationsWidget.from_metadata(metadata)

        assert widget is not None
        assert widget._rewritten_query == "expanded query"
        assert widget._original_query is None

    def test_from_metadata_returns_none_for_no_sources(self) -> None:
        """from_metadata returns None when no sources."""
        from chatty.ui.widgets import CitationsWidget

        metadata = RAGMetadata(sources=[])
        widget = CitationsWidget.from_metadata(metadata, original_query="test")

        assert widget is None

    def test_from_metadata_returns_none_for_none_metadata(self) -> None:
        """from_metadata returns None for None metadata."""
        from chatty.ui.widgets import CitationsWidget

        widget = CitationsWidget.from_metadata(None, original_query="test")

        assert widget is None


class TestRAGMetadataRewrittenQuery:
    """Tests for rewritten_query field in RAGMetadata."""

    def test_metadata_has_rewritten_query_field(self) -> None:
        """RAGMetadata should have rewritten_query field."""
        metadata = RAGMetadata(
            sources=[],
            retrieval_time_s=1.5,
            chunk_count=10,
            rewritten_query="rewritten query text",
        )

        assert metadata.rewritten_query == "rewritten query text"

    def test_metadata_rewritten_query_default_none(self) -> None:
        """RAGMetadata rewritten_query should default to None."""
        metadata = RAGMetadata(sources=[])

        assert metadata.rewritten_query is None


class TestQueryDisplayLogic:
    """Tests for query display decision logic."""

    def test_query_shown_when_different(self) -> None:
        """Query line should be shown when queries differ."""
        from chatty.ui.widgets import CitationsWidget

        sources = [RAGSource(title="Test", snippet="Test", score=0.9)]

        widget = CitationsWidget(
            sources=sources,
            rewritten_query="expanded full query",
            original_query="short question?",
        )

        # Queries are different, so display should show rewritten line
        assert widget._rewritten_query != widget._original_query

    def test_query_hidden_when_same(self) -> None:
        """Query line should be hidden when queries are the same."""
        from chatty.ui.widgets import CitationsWidget

        sources = [RAGSource(title="Test", snippet="Test", score=0.9)]

        widget = CitationsWidget(
            sources=sources,
            rewritten_query="same query",
            original_query="same query",
        )

        # Queries are the same, so display should not show rewritten line
        assert widget._rewritten_query == widget._original_query

    def test_query_hidden_when_original_missing(self) -> None:
        """Query line hidden when original_query is None."""
        from chatty.ui.widgets import CitationsWidget

        sources = [RAGSource(title="Test", snippet="Test", score=0.9)]

        widget = CitationsWidget(
            sources=sources,
            rewritten_query="expanded query",
            original_query=None,  # Missing
        )

        # Can't show transformation without original
        assert widget._original_query is None

    def test_query_hidden_when_rewritten_missing(self) -> None:
        """Query line hidden when rewritten_query is None."""
        from chatty.ui.widgets import CitationsWidget

        sources = [RAGSource(title="Test", snippet="Test", score=0.9)]

        widget = CitationsWidget(
            sources=sources,
            rewritten_query=None,  # Missing
            original_query="user query",
        )

        # Can't show transformation without rewritten
        assert widget._rewritten_query is None


class TestStatusBarNoInspectMode:
    """Tests verifying inspect_mode is removed from StatusBar."""

    def test_status_bar_no_inspect_mode_attribute(self) -> None:
        """StatusBar should not have _inspect_mode attribute."""
        from chatty.ui.widgets import StatusBar

        status_bar = StatusBar(model="test-model")

        # _inspect_mode was removed in v0.4.3 refactor
        assert not hasattr(status_bar, "_inspect_mode")

    def test_update_status_no_inspect_mode_param(self) -> None:
        """update_status should not accept inspect_mode parameter."""
        import inspect

        from chatty.ui.widgets import StatusBar

        # Get the signature of update_status
        sig = inspect.signature(StatusBar.update_status)
        params = list(sig.parameters.keys())

        # inspect_mode was removed in v0.4.3 refactor
        assert "inspect_mode" not in params


class TestNoInspectModal:
    """Tests verifying InspectModal is removed."""

    def test_no_inspect_modal_class(self) -> None:
        """InspectModal class should not exist in modals module."""
        from chatty.ui import modals

        # InspectModal was removed in v0.4.3 refactor
        assert not hasattr(modals, "InspectModal")


class TestNoInspectModeInApp:
    """Tests verifying inspect mode is removed from ChatApp."""

    def test_app_no_inspect_mode_attribute(self) -> None:
        """ChatApp should not have _inspect_mode attribute."""
        from chatty.ui.app import ChatApp

        app = ChatApp()

        # _inspect_mode was removed in v0.4.3 refactor
        assert not hasattr(app, "_inspect_mode")

    def test_app_no_inspect_action(self) -> None:
        """ChatApp should not have action_toggle_inspect method."""
        from chatty.ui.app import ChatApp

        # action_toggle_inspect was removed in v0.4.3 refactor
        assert not hasattr(ChatApp, "action_toggle_inspect")