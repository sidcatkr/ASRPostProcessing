import unittest

from asrpostprocessing.text import make_character_diff_html, make_diff_export_document, make_diff_html


class InlineDiffHtmlTest(unittest.TestCase):
    def test_make_diff_html_uses_inline_flow_instead_of_table_rows(self):
        html = make_diff_html("불련 코드\n기존 문장", "Boolean 코드\n기존 문장")

        self.assertIn("asrpp-inline-diff", html)
        self.assertIn("asrpp-diff-text", html)
        self.assertIn("Boolean", html)
        self.assertIn("asrpp-diff-delete", html)
        self.assertIn("asrpp-diff-insert", html)
        self.assertNotIn("<table", html.lower())
        self.assertNotIn("diff_next", html)

    def test_make_diff_html_escapes_transcript_content(self):
        html = make_diff_html("<raw>", "<corrected>")

        self.assertIn("&lt;", html)
        self.assertIn("&gt;", html)
        self.assertNotIn("<raw>", html)
        self.assertNotIn("<corrected>", html)

    def test_make_diff_html_supports_reference_labels_and_no_change_state(self):
        html = make_diff_html("정답 문장", "정답 문장", "Reference", "Corrected")

        self.assertIn("Reference", html)
        self.assertIn("Corrected", html)
        self.assertIn("No character changes", html)
        self.assertIn("No deletion, insertion, or replacement was detected", html)
        self.assertIn("var(--block-background-fill", html)
        self.assertIn("var(--body-text-color", html)
        self.assertIn("var(--block-radius", html)

    def test_make_diff_html_explains_change_types(self):
        html = make_diff_html("A B C D", "A X C", "Raw", "Corrected")

        self.assertIn("Deleted", html)
        self.assertIn("Inserted", html)
        self.assertIn("Replaced", html)
        self.assertIn("Change details", html)
        self.assertIn("Deletion", html)
        self.assertIn("Removed from Raw", html)
        self.assertIn("Replacement", html)
        self.assertIn("Changed Raw text into Corrected text", html)
        self.assertIn("Raw:</span> <code>B", html)
        self.assertIn("Corrected:</span> <code>X", html)

    def test_make_diff_export_document_wraps_fragment_for_report_use(self):
        fragment = make_diff_html("A B", "A X", "Raw", "Corrected")
        html = make_diff_export_document(fragment, title="Report Diff", metadata={"Run ID": "run-1"})

        self.assertIn("<!doctype html>", html)
        self.assertIn("<title>Report Diff</title>", html)
        self.assertIn("Run ID", html)
        self.assertIn("run-1", html)
        self.assertIn("Change details", html)
        self.assertIn("@media print", html)

    def test_make_diff_html_can_show_cer_wer_error_monitor(self):
        html = make_diff_html("Alpha Term 실행", "Alpha 실행", "Reference", "Corrected", show_error_monitor=True)

        self.assertIn("CER/WER error monitor", html)
        self.assertIn("Metric location", html)
        self.assertIn("Rate part", html)
        self.assertIn("Error share", html)
        self.assertIn("Reference context", html)
        self.assertIn("Hypothesis context", html)
        self.assertIn("without spacing, line breaks, punctuation, or symbols", html)
        self.assertIn("Term", html)
        self.assertIn("[term]", html.lower())

    def test_make_character_diff_html_marks_small_character_changes(self):
        html = make_character_diff_html("앞 문장 불련 코드 뒤 문장", "앞 문장 Boolean 코드 뒤 문장")

        self.assertIn("asrpp-diff-delete", html)
        self.assertIn("asrpp-diff-insert", html)
        self.assertIn("Boolean", html)
        self.assertIn("앞 문장", html)
        self.assertIn("뒤 문장", html)


if __name__ == "__main__":
    unittest.main()
