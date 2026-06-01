import unittest

from asrpostprocessing.text import make_diff_html


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


if __name__ == "__main__":
    unittest.main()
