"""Readiness reports must distinguish success, failure and skipped checks."""
from tools.check_install import summarize


def test_component_summary_does_not_hide_failures_or_skips(tmp_path):
    report = tmp_path / 'checks.xml'
    report.write_text('''<testsuites><testsuite>
      <testcase classname="tests.core.test_energy" name="energy"/>
      <testcase classname="tests.inference.test_predict" name="prediction"><failure/></testcase>
      <testcase classname="tests.test_batch_failures" name="training"/>
      <testcase classname="tests.test_batch_failures" name="conversion"><skipped/></testcase>
      <testcase classname="tests.incident_response.test_workflows" name="workflows"/>
    </testsuite></testsuites>''')
    groups = summarize(report)
    assert groups['Detection energy and voting'] == {'verified', 'failed'}
    assert groups['Training and conversion control flow'] == {'verified', 'skipped'}
    assert groups['Incident response workflow selection'] == {'verified'}
