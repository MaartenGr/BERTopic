import copy
import pytest
import numpy as np
import pandas as pd


@pytest.mark.parametrize(
    "model",
    [
        "base_topic_model",
        "custom_topic_model",
        "reduced_topic_model",
    ],
)
class TestCreateTopicTaxonomy:
    """Tests for the create_topic_taxonomy method."""

    def _get_model_and_nr_topics(self, model, request):
        """Get a deep copy of the model and count non-outlier topics."""
        topic_model = copy.deepcopy(request.getfixturevalue(model))
        nr_topics = len([t for t in topic_model.topic_sizes_.keys() if t != -1])
        return topic_model, nr_topics

    def test_basic_two_levels(self, model, documents, request):
        """Test basic two-level taxonomy creation."""
        topic_model, nr_topics = self._get_model_and_nr_topics(model, request)
        if nr_topics < 4:
            pytest.skip("Not enough topics for two-level taxonomy")

        has_outliers = topic_model._outliers == 1
        level_1 = max(nr_topics // 3, 2)
        level_2 = max(level_1 // 3, 2)
        if level_2 >= level_1:
            level_2 = level_1 - 1

        hierarchy = topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[level_1, level_2])

        assert isinstance(hierarchy, pd.DataFrame)
        assert set(hierarchy.columns) == {
            "Topic_ID",
            "Topic_Name",
            "Level",
            "Parent_ID",
            "Parent_Name",
            "Document_Count",
        }

        # Check levels exist
        assert set(hierarchy.Level.unique()) == {0, 1, 2}

        # Leaf topics include non-outlier topics (and outlier if present)
        leaves = hierarchy[hierarchy.Level == 0]
        expected_leaf_ids = sorted([t for t in topic_model.topic_sizes_.keys() if t != -1])
        if has_outliers:
            expected_leaf_ids = [-1, *expected_leaf_ids]
        assert sorted(leaves.Topic_ID.tolist()) == expected_leaf_ids

        # Top-level topics have Parent_ID == -2
        top_level = hierarchy[hierarchy.Level == 2]
        assert (top_level.Parent_ID == -2).all()

        # Non-top-level topics have valid parent references
        for _, row in hierarchy[hierarchy.Level < 2].iterrows():
            parent_exists = (hierarchy.Topic_ID == row.Parent_ID).any()
            assert parent_exists, f"Topic {row.Topic_ID} references non-existent parent {row.Parent_ID}"

    def test_single_level(self, model, documents, request):
        """Test single-level taxonomy (just parents above leaves)."""
        topic_model, nr_topics = self._get_model_and_nr_topics(model, request)
        if nr_topics < 3:
            pytest.skip("Not enough topics")

        target = max(nr_topics // 2, 2)
        hierarchy = topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[target])

        assert set(hierarchy.Level.unique()) == {0, 1}
        top_level = hierarchy[hierarchy.Level == 1]
        assert (top_level.Parent_ID == -2).all()

    def test_document_count_consistency(self, model, documents, request):
        """Test that document counts sum correctly up the hierarchy (non-outlier parents)."""
        topic_model, nr_topics = self._get_model_and_nr_topics(model, request)
        if nr_topics < 4:
            pytest.skip("Not enough topics")

        level_1 = max(nr_topics // 2, 2)
        hierarchy = topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[level_1])

        # For each non-outlier parent, doc count should equal the sum of children's doc counts
        parents = hierarchy[(hierarchy.Level == 1) & (hierarchy.Topic_Name != "Outlier")]
        for _, parent in parents.iterrows():
            children = hierarchy[hierarchy.Parent_ID == parent.Topic_ID]
            assert parent.Document_Count == children.Document_Count.sum(), (
                f"Parent {parent.Topic_ID} doc count {parent.Document_Count} != "
                f"children sum {children.Document_Count.sum()}"
            )

    def test_min_children_enforced(self, model, documents, request):
        """Test that min_children constraint is enforced for non-outlier parents."""
        topic_model, nr_topics = self._get_model_and_nr_topics(model, request)
        if nr_topics < 6:
            pytest.skip("Not enough topics for min_children test")

        target = max(nr_topics // 2, 3)
        hierarchy = topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[target], min_children=2)

        # Every non-outlier parent must have at least 2 children
        parents = hierarchy[(hierarchy.Level == 1) & (hierarchy.Topic_Name != "Outlier")]
        for _, parent in parents.iterrows():
            n_children = (hierarchy.Parent_ID == parent.Topic_ID).sum()
            assert n_children >= 2, f"Parent {parent.Topic_ID} has only {n_children} children"

    def test_min_children_three(self, model, documents, request):
        """Test min_children=3 enforcement for non-outlier parents."""
        topic_model, nr_topics = self._get_model_and_nr_topics(model, request)
        if nr_topics < 9:
            pytest.skip("Not enough topics for min_children=3 test")

        target = max(nr_topics // 3, 3)
        hierarchy = topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[target], min_children=3)

        parents = hierarchy[(hierarchy.Level == 1) & (hierarchy.Topic_Name != "Outlier")]
        for _, parent in parents.iterrows():
            n_children = (hierarchy.Parent_ID == parent.Topic_ID).sum()
            assert n_children >= 3, f"Parent {parent.Topic_ID} has only {n_children} children (expected >= 3)"

    def test_with_embeddings(self, model, documents, document_embeddings, request):
        """Test taxonomy creation with document embeddings."""
        topic_model, nr_topics = self._get_model_and_nr_topics(model, request)
        if nr_topics < 3:
            pytest.skip("Not enough topics")

        has_outliers = topic_model._outliers == 1
        expected_leaves = nr_topics + (1 if has_outliers else 0)

        target = max(nr_topics // 2, 2)
        hierarchy = topic_model.create_topic_taxonomy(
            documents,
            nr_topics_per_level=[target],
            embeddings=document_embeddings,
            doc_embedding_weight=0.7,
        )

        assert isinstance(hierarchy, pd.DataFrame)
        assert len(hierarchy[hierarchy.Level == 0]) == expected_leaves

    def test_use_ctfidf(self, model, documents, request):
        """Test taxonomy creation using c-TF-IDF for distance computation."""
        topic_model, nr_topics = self._get_model_and_nr_topics(model, request)
        if nr_topics < 3:
            pytest.skip("Not enough topics")

        target = max(nr_topics // 2, 2)
        hierarchy = topic_model.create_topic_taxonomy(
            documents,
            nr_topics_per_level=[target],
            use_ctfidf=True,
        )

        assert isinstance(hierarchy, pd.DataFrame)
        assert set(hierarchy.Level.unique()) == {0, 1}

    def test_every_leaf_has_parent(self, model, documents, request):
        """Test that every leaf topic is assigned exactly one parent."""
        topic_model, nr_topics = self._get_model_and_nr_topics(model, request)
        if nr_topics < 3:
            pytest.skip("Not enough topics")

        target = max(nr_topics // 2, 2)
        hierarchy = topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[target])

        leaves = hierarchy[hierarchy.Level == 0]
        # Every leaf must have a non-null Parent_ID that is not -2
        assert leaves.Parent_ID.notna().all()
        assert (leaves.Parent_ID != -2).all()

    def test_no_side_effects(self, model, documents, request):
        """Test that create_topic_taxonomy does not modify the fitted model."""
        topic_model, nr_topics = self._get_model_and_nr_topics(model, request)
        if nr_topics < 3:
            pytest.skip("Not enough topics")

        topics_before = topic_model.topics_.copy()
        sizes_before = dict(topic_model.topic_sizes_)
        embeddings_before = topic_model.topic_embeddings_.copy()

        target = max(nr_topics // 2, 2)
        topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[target])

        assert topic_model.topics_ == topics_before
        assert dict(topic_model.topic_sizes_) == sizes_before
        np.testing.assert_array_equal(topic_model.topic_embeddings_, embeddings_before)

    def test_topic_names_not_empty(self, model, documents, request):
        """Test that all topics have non-empty names."""
        topic_model, nr_topics = self._get_model_and_nr_topics(model, request)
        if nr_topics < 3:
            pytest.skip("Not enough topics")

        target = max(nr_topics // 2, 2)
        hierarchy = topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[target])

        assert (hierarchy.Topic_Name.str.len() > 0).all()
        # Parent names should also be populated for non-top-level
        non_top = hierarchy[hierarchy.Parent_ID != -2]
        assert (non_top.Parent_Name.str.len() > 0).all()

    def test_outlier_chain(self, model, documents, request):
        """Test that outlier topics form a single-child chain through all levels."""
        topic_model, nr_topics = self._get_model_and_nr_topics(model, request)
        if nr_topics < 4:
            pytest.skip("Not enough topics")
        if topic_model._outliers != 1:
            pytest.skip("No outlier topic in this model")

        level_1 = max(nr_topics // 3, 2)
        level_2 = max(level_1 // 3, 2)
        if level_2 >= level_1:
            level_2 = level_1 - 1

        hierarchy = topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[level_1, level_2])

        # Outlier leaf (-1) should exist
        outlier_leaf = hierarchy[(hierarchy.Level == 0) & (hierarchy.Topic_ID == -1)]
        assert len(outlier_leaf) == 1

        # Walk up the chain: each outlier parent should have exactly 1 child (the outlier below)
        current_id = outlier_leaf.iloc[0].Parent_ID
        for level in range(1, 3):
            outlier_node = hierarchy[(hierarchy.Level == level) & (hierarchy.Topic_ID == current_id)]
            assert len(outlier_node) == 1
            assert outlier_node.iloc[0].Topic_Name == "Outlier"
            n_children = (hierarchy.Parent_ID == current_id).sum()
            assert n_children == 1  # outlier parents always have exactly 1 child
            current_id = outlier_node.iloc[0].Parent_ID

        # Top-level outlier should point to Root
        assert current_id == -2


class TestCreateTopicTaxonomyValidation:
    """Tests for parameter validation."""

    def test_invalid_nr_topics_per_level_empty(self, base_topic_model, documents):
        topic_model = copy.deepcopy(base_topic_model)
        with pytest.raises(ValueError, match="non-empty list"):
            topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[])

    def test_invalid_nr_topics_per_level_too_large(self, base_topic_model, documents):
        topic_model = copy.deepcopy(base_topic_model)
        nr_topics = len([t for t in topic_model.topic_sizes_.keys() if t != -1])
        with pytest.raises(ValueError, match="must be less than"):
            topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[nr_topics + 1])

    def test_invalid_nr_topics_per_level_zero(self, base_topic_model, documents):
        topic_model = copy.deepcopy(base_topic_model)
        with pytest.raises(ValueError, match=">= 1"):
            topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[0])

    def test_invalid_min_children(self, base_topic_model, documents):
        topic_model = copy.deepcopy(base_topic_model)
        with pytest.raises(ValueError, match="min_children"):
            topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[2], min_children=0)

    def test_invalid_doc_embedding_weight(self, base_topic_model, documents):
        topic_model = copy.deepcopy(base_topic_model)
        with pytest.raises(ValueError, match="doc_embedding_weight"):
            topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[2], doc_embedding_weight=1.5)

    def test_invalid_level_sequence(self, base_topic_model, documents):
        """Test that level k+1 target must be less than level k target."""
        topic_model = copy.deepcopy(base_topic_model)
        nr_topics = len([t for t in topic_model.topic_sizes_.keys() if t != -1])
        if nr_topics < 6:
            pytest.skip("Not enough topics")
        # level 1 = 3, level 2 = 5 is invalid (5 >= 3)
        with pytest.raises(ValueError, match="must be less than"):
            topic_model.create_topic_taxonomy(documents, nr_topics_per_level=[3, 5])
