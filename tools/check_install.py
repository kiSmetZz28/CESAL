"""Present the software checks by component, with diagnostics saved to disk."""
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUPS = {
    'Checkpoint download and archive integrity': ('test_checkpoint_downloads',),
    'Detection energy and voting': ('test_energy', 'test_voting', 'test_predict'),
    'EM-GMM threshold calibration': ('test_threshold',),
    'Pipeline execution and failure handling': ('test_pipeline_failures', 'test_sweep'),
    'Training and conversion control flow': ('test_batch_failures',),
    'Progress and metric reporting': ('test_steps',),
    'Incident response workflow selection': ('test_workflows',),
    'Small-experiment inputs and readiness reporting': ('test_smoke', 'test_readiness'),
}


def summarize(report):
    """Return component outcomes, including skipped checks rather than hiding them."""
    groups = {}
    for case in ET.parse(report).iter('testcase'):
        module = case.get('classname', '').split('.')[-1]
        label = next((label for label, modules in GROUPS.items() if module in modules),
                     'Additional software checks')
        state = ('failed' if case.find('failure') is not None or case.find('error') is not None
                 else 'skipped' if case.find('skipped') is not None else 'verified')
        groups.setdefault(label, set()).add(state)
    return groups


def main():
    base = ROOT / 'outputs' / 'checks'
    base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run_', dir=base))
    report, log = output / 'checks.xml', output / 'checks.log'
    print('Checking CESAL software components…', flush=True)
    with log.open('w') as stream:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider',
             f'--junitxml={report}'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
        )
    groups = summarize(report) if report.exists() else {}
    for label, states in groups.items():
        status = ('NEEDS ATTENTION' if 'failed' in states else
                  'PARTLY VERIFIED (optional checks skipped)' if states == {'verified', 'skipped'} else
                  'NOT CHECKED' if states == {'skipped'} else 'VERIFIED')
        print(f'  {status} — {label}')
    print(f'\nDetailed diagnostics: {log.relative_to(ROOT)}')
    if result.returncode or not groups:
        print('Setup checks did not complete successfully. Review the diagnostics above.')
        return result.returncode or 1
    print('Core software checks completed successfully.')
    print('Next: python run.py smoke os — verify real edge and cloud model execution.')
    print('LLM classification is checked separately with python run.py classify <model>.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
