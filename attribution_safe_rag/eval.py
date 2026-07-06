from __future__ import annotations

import argparse
import re
from typing import Any

try:
    from answer import build_grounded_answer
    from config import ConfigError, load_settings
    from context_budget import format_contexts_for_prompt, select_contexts_for_prompt
    from router import detect_query_type
    from retrieve import retrieve_evidence, expand_to_parent_context
    from verifier import (
        check_metadata_consistency,
        detect_query_entity,
        verify_entity_ownership,
    )
except ImportError:
    from .answer import build_grounded_answer
    from .config import ConfigError, load_settings
    from .context_budget import format_contexts_for_prompt, select_contexts_for_prompt
    from .router import detect_query_type
    from .retrieve import retrieve_evidence, expand_to_parent_context
    from .verifier import (
        check_metadata_consistency,
        detect_query_entity,
        verify_entity_ownership,
    )


TEST_CASES = [
    {
        "query": "Which resume has the Fraud Detection System project?",
        "expected_query_type": "ownership",
        "expected_owner": "Candidate A",
        "expected_entity": "Fraud Detection System",
        "expected_source_filename": "resume_a.txt",
    },
    {
        "query": "Who built the Credit Risk Dashboard?",
        "expected_query_type": "ownership",
        "expected_owner": "Candidate B",
        "expected_entity": "Credit Risk Dashboard",
        "expected_source_filename": "resume_b.txt",
    },
    {
        "query": "Where is SSO setup explained?",
        "expected_query_type": "procedure",
        "expected_document_type": "support_doc",
        "expected_entity": "SSO Setup",
        "expected_source_filename": "sample_support_doc.txt",
    },
    {
        "query": "How do I configure SSO?",
        "expected_query_type": "procedure",
        "expected_document_type": "support_doc",
        "expected_entity": "SSO Setup",
        "expected_source_filename": "sample_support_doc.txt",
    },
    {
        "query": "Compare Candidate A and Candidate B",
        "expected_query_type": "comparison",
        "expected_source_filenames": ["resume_a.txt", "resume_b.txt"],
    },
    {
        "query": "What is the difference between SSO Setup and Dashboard Login?",
        "expected_query_type": "comparison",
        "expected_source_filenames": ["sample_support_doc.txt"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate attribution-safe RAG behavior.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence units to retrieve.")
    parser.add_argument("--workspace-id", default=None, help="Workspace to evaluate.")
    return parser.parse_args()


def contains_expected_entity(detected_entity: str | None, expected_entity: str | None) -> bool:
    if not expected_entity:
        return True
    if not detected_entity:
        return False
    return expected_entity.lower() in detected_entity.lower() or detected_entity.lower() in expected_entity.lower()


def comparison_answer_is_valid(answer: str, expected_sources: list[str]) -> bool:
    normalized = answer.lower()
    has_structure = "source a:" in normalized and "source b:" in normalized and "differences:" in normalized
    if not has_structure:
        return False
    return all(source.lower() in normalized for source in expected_sources)


def evaluate_test_case(
    test_case: dict[str, Any],
    workspace_id: str,
    top_k: int,
) -> dict[str, Any]:
    query = test_case["query"]
    query_type = detect_query_type(query)
    evidence_items = retrieve_evidence(query, top_k, workspace_id, scope="seeded")
    parent_contexts = expand_to_parent_context(evidence_items)
    compact_contexts = select_contexts_for_prompt(parent_contexts)
    consistency = check_metadata_consistency(evidence_items)
    detected_entity = detect_query_entity(query, evidence_items)
    ownership_result = (
        verify_entity_ownership(evidence_items, detected_entity) if detected_entity else None
    )
    final_answer, _, _, _ = build_grounded_answer(query, evidence_items)

    retrieved_sources = [
        item.get("metadata", {}).get("source_filename")
        for item in evidence_items
        if item.get("metadata", {}).get("source_filename")
    ]
    retrieved_document_types = [
        item.get("metadata", {}).get("document_type")
        for item in evidence_items
        if item.get("metadata", {}).get("document_type")
    ]

    if "expected_source_filename" in test_case:
        retrieval_hit = test_case["expected_source_filename"] in retrieved_sources
    else:
        retrieval_hit = all(
            expected_source in retrieved_sources
            for expected_source in test_case.get("expected_source_filenames", [])
        )
    query_type_correct = query_type == test_case.get("expected_query_type")
    entity_detected_correctly = contains_expected_entity(
        detected_entity, test_case.get("expected_entity")
    )

    expected_owner = test_case.get("expected_owner")
    ownership_correct = True
    wrong_attribution = False

    if expected_owner:
        verified_names = ownership_result.get("candidate_names", []) if ownership_result else []
        ownership_correct = expected_owner in verified_names
        if ownership_result and ownership_result.get("ownership_is_unambiguous") and not ownership_correct:
            wrong_attribution = True

    expected_document_type = test_case.get("expected_document_type")
    if expected_document_type:
        ownership_correct = expected_document_type in retrieved_document_types
        if "belongs to" in final_answer.lower():
            wrong_attribution = True

    comparison_correct = True
    if query_type == "comparison":
        comparison_correct = comparison_answer_is_valid(
            final_answer,
            test_case.get("expected_source_filenames", []),
        )

    answer_grounded = (
        "Supported by evidence" in final_answer
        or final_answer.strip().lower() == "insufficient evidence"
        or re.search(r"\bevi_[a-z0-9]+\b", final_answer) is not None
    )
    context_budget_used = len(format_contexts_for_prompt(compact_contexts)) <= 3500 and bool(compact_contexts)
    passed = retrieval_hit and not wrong_attribution
    if expected_owner:
        passed = passed and ownership_correct
    if expected_document_type:
        passed = passed and expected_document_type in retrieved_document_types
    passed = passed and query_type_correct and answer_grounded and comparison_correct
    if test_case.get("expected_entity"):
        passed = passed and entity_detected_correctly

    return {
        "query": query,
        "expected": test_case,
        "query_type": query_type,
        "retrieved_sources": retrieved_sources,
        "retrieved_document_types": retrieved_document_types,
        "detected_entity": detected_entity,
        "consistency": consistency,
        "ownership_result": ownership_result,
        "final_answer": final_answer,
        "query_type_correct": query_type_correct,
        "retrieval_hit": retrieval_hit,
        "ownership_correct": ownership_correct,
        "wrong_attribution": wrong_attribution,
        "comparison_correct": comparison_correct,
        "answer_grounded": answer_grounded,
        "context_budget_used": context_budget_used,
        "passed": passed,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_tests = len(results)
    passed_count = sum(1 for result in results if result["passed"])
    failed_count = total_tests - passed_count
    wrong_attribution_count = sum(1 for result in results if result["wrong_attribution"])
    retrieval_hit_count = sum(1 for result in results if result["retrieval_hit"])
    query_type_correct_count = sum(1 for result in results if result["query_type_correct"])
    ownership_correct_count = sum(
        1
        for result in results
        if result.get("expected", {}).get("expected_owner") is None or result["ownership_correct"]
    )
    comparison_correct_count = sum(1 for result in results if result["comparison_correct"])

    return {
        "total_tests": total_tests,
        "passed": passed_count,
        "failed": failed_count,
        "wrong_attribution_rate": wrong_attribution_count / total_tests if total_tests else 0.0,
        "retrieval_hit_rate": retrieval_hit_count / total_tests if total_tests else 0.0,
        "query_type_accuracy": query_type_correct_count / total_tests if total_tests else 0.0,
        "ownership_accuracy": ownership_correct_count / total_tests if total_tests else 0.0,
        "comparison_accuracy": comparison_correct_count / total_tests if total_tests else 0.0,
    }


def run_benchmark(workspace_id: str, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = [
        evaluate_test_case(test_case, workspace_id, top_k)
        for test_case in TEST_CASES
    ]
    return results, summarize_results(results)


def print_report(results: list[dict[str, Any]]) -> None:
    print("Evaluation Results")
    print("------------------")

    for result in results:
        print(f"Query: {result['query']}")
        print(f"Expected: {result['expected']}")
        print(f"Query type: {result['query_type']}")
        print(f"Retrieved sources: {result['retrieved_sources']}")
        print(f"Detected entity: {result['detected_entity']}")
        print(f"Final answer: {result['final_answer']}")
        print(f"query_type_correct: {str(result['query_type_correct']).lower()}")
        print(f"retrieval_hit: {str(result['retrieval_hit']).lower()}")
        print(f"ownership_correct: {str(result['ownership_correct']).lower()}")
        print(f"wrong_attribution: {str(result['wrong_attribution']).lower()}")
        print(f"comparison_correct: {str(result['comparison_correct']).lower()}")
        print(f"answer_grounded: {str(result['answer_grounded']).lower()}")
        print(f"context_budget_used: {str(result['context_budget_used']).lower()}")
        print(f"passed: {str(result['passed']).lower()}")
        print()

    summary = summarize_results(results)

    print("Summary")
    print("-------")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Wrong attribution rate: {summary['wrong_attribution_rate']:.2%}")
    print(f"Retrieval hit rate: {summary['retrieval_hit_rate']:.2%}")
    print(f"Query type accuracy: {summary['query_type_accuracy']:.2%}")
    print(f"Ownership accuracy: {summary['ownership_accuracy']:.2%}")
    print(f"Comparison accuracy: {summary['comparison_accuracy']:.2%}")


if __name__ == "__main__":
    args = parse_args()
    try:
        settings = load_settings()
        workspace_id = args.workspace_id or settings.default_workspace_id
        results, _ = run_benchmark(workspace_id, args.top_k)
        print_report(results)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
